# === M3_RDE_GUI.py (Upgraded Reflective Logic Only) ===
# GUI phản chiếu RDE với navigation (Next/Prev) và lưu kết quả M3_results.json
# === Nâng cấp 2025-07-16 + 2025-07-17: Tích hợp output từ M2 vào reflective logic ===

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import tkinter as tk
from tkinter import filedialog, scrolledtext
import json
import faiss
import pickle
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from datetime import datetime

# === Load models & concept library ===
index = faiss.read_index("concept_index.faiss")
embedding_dim = index.d

if embedding_dim == 384:
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
elif embedding_dim == 768:
    model_name = "sentence-transformers/paraphrase-mpnet-base-v2"
else:
    raise ValueError(f"Unsupported FAISS embedding dim: {embedding_dim}")

print(f"[RDE] Loading model: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

with open("concept_metadata.pkl", "rb") as f:
    concept_lib = pickle.load(f)

# === Load Divergence Type Definitions & Vectors ===
with open("divergence_types_definitions.json", "r", encoding="utf-8") as f:
    divergence_definitions = json.load(f)

with open("divergence_types_vectors.pkl", "rb") as f:
    divergence_vectors = pickle.load(f)  # Dict[str, np.ndarray]

# === Threshold table (unchanged) ===
threshold_table = {
    "Factual": {"angle_thresh": 20, "drift_thresh": 0.1, "lexical_min": 0.75},
    "Emotional": {"angle_thresh": 35, "drift_thresh": 0.25, "lexical_min": 0.2},
    "Reflective": {"angle_thresh": 30, "drift_thresh": 0.2, "lexical_min": 0.3},
    "Instructional": {"angle_thresh": 25, "drift_thresh": 0.15, "lexical_min": 0.6},
    "Existential": {"angle_thresh": 40, "drift_thresh": 0.3, "lexical_min": 0.15},
    "Conversational": {"angle_thresh": 35, "drift_thresh": 0.2, "lexical_min": 0.3}
}

# === Embedding & Metrics ===
def embed_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)[0].cpu().numpy()

def cosine_distance(a, b):
    return 1 - (np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def lexical_overlap(a, b):
    set1, set2 = set(a.lower().split()), set(b.lower().split())
    return len(set1.intersection(set2)) / max(len(set1), 1)

def find_nearest_concepts(vec, k=3):
    vec = np.array([vec]).astype("float32")
    D, I = index.search(vec, k)
    return [(concept_lib[i], float(D[0][j]), i) for j, i in enumerate(I[0])]

def drift_angle(a, b):
    cosine_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return np.arccos(np.clip(cosine_sim, -1.0, 1.0)) * 180 / np.pi

def logic_trajectory_drift(prompt_vec, response_vec, anchor_vec):
    path_vec = anchor_vec - prompt_vec
    proj = np.dot(response_vec - prompt_vec, path_vec) / np.linalg.norm(path_vec)
    proj_point = prompt_vec + proj * path_vec / np.linalg.norm(path_vec)
    return np.linalg.norm(response_vec - proj_point)

def reflective_divergence_scan(prompt, response, m2_entry=None):
    divergence_types = []
    vec_prompt = embed_text(prompt)
    vec_response = embed_text(response)
    lexical_sim = lexical_overlap(prompt, response)

    nearest = find_nearest_concepts(vec_response, k=3)
    concept_vecs = [embed_text(c['term']) for c, _, _ in nearest]
    anchor_vec = np.mean(concept_vecs, axis=0)

    angle = drift_angle(vec_prompt, vec_response)
    ltd = logic_trajectory_drift(vec_prompt, vec_response, anchor_vec)

    intent = "Conversational"
    th = threshold_table.get(intent, threshold_table["Conversational"])

    # === Nếu có dữ liệu từ M2 ===
    glitch_types = []
    csi_score = 0.0
    srl_score = 0.0
    route_flags_m2 = set()
    if m2_entry:
        m2_results = m2_entry.get("results", {})
        glitch_types = m2_results.get("glitch_types", [])
        csi_score = m2_results.get("csi_score", 0.0)
        srl_score = m2_results.get("aggregated_srl_score", 0.0)
        if m2_results.get("route_sahl"): route_flags_m2.add("sahl")
        if m2_results.get("route_arp_x"): route_flags_m2.add("arp_x")
        if m2_results.get("route_zelc"): route_flags_m2.add("zelc")

    # === So khớp divergence types ===
    score_map = {}
    routing_flags = set(route_flags_m2)
    for div_type, definition in divergence_definitions.items():
        vec_type = divergence_vectors.get(div_type)
        if vec_type is None:
            continue
        similarity = 1 - cosine_distance(vec_response, vec_type)
        conditions_met = (
            similarity > 0.88 or
            (definition.get("short_description") and similarity > 0.82 and lexical_sim < 0.4)
        )
        if conditions_met or div_type in glitch_types:
            divergence_types.append(div_type)
            score_map[div_type] = definition.get("score_contribution", 0.1)
            routing_flags.update(definition.get("trigger_modules", []))

    dvg_signature = "DVG::" + "+".join(divergence_types) if divergence_types else "DVG::None"

    ds_vec = min(sum(score_map.get(t, 0.1) for t in divergence_types), 1.0)
    ds = round((ds_vec + csi_score) / 2, 3) if csi_score > 0 else round(ds_vec, 3)
    re_base = 0.1 * len(divergence_types) + lexical_sim * 0.2
    re = round(min(re_base + srl_score * 0.5, 1.0), 3)

    return {
        "dvg_signature": dvg_signature,
        "divergence_types": divergence_types,
        "divergence_score": ds,
        "reflective_entropy": re,
        "route_sahl": "sahl" in routing_flags or re > 0.6,
        "route_arp_x": "arp_x" in routing_flags or ds > 0.6,
        "route_zelc": "zelc" in routing_flags or re > 0.65,
        "zelc_tier2": "zelc_tier2" in routing_flags,
        "entropy_time_trigger": "entropy_time_trigger" in routing_flags,
        "entropy_anomaly_trigger": "entropy_anomaly_trigger" in routing_flags
    }

# === GUI + Navigation Class ===
class RDE_GUI:
    def __init__(self, master):
        self.master = master
        master.title("Reflective Divergence Engine (M3)")
        master.geometry("840x580")

        self.load_button = tk.Button(master, text="Load JSON", command=self.load_json)
        self.load_button.pack()

        self.textbox = scrolledtext.ScrolledText(master, wrap=tk.WORD, width=100, height=30)
        self.textbox.pack(padx=10, pady=10)

        nav_frame = tk.Frame(master)
        nav_frame.pack()
        self.prev_button = tk.Button(nav_frame, text="Previous", command=self.prev_entry)
        self.prev_button.pack(side=tk.LEFT, padx=5)
        self.next_button = tk.Button(nav_frame, text="Next", command=self.next_entry)
        self.next_button.pack(side=tk.LEFT, padx=5)

        self.entries = []
        self.current_index = 0

    def load_json(self):
        path = filedialog.askopenfilename(filetypes=[("JSON Files", "*.json")])
        if path:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    self.entries = data
                elif isinstance(data, dict):
                    if "entries" in data:
                        self.entries = data["entries"]
                    else:
                        self.entries = [data]
                else:
                    self.entries = []
            self.current_index = 0
            self.run_and_show()

    def run_and_show(self):
        if 0 <= self.current_index < len(self.entries):
            entry = self.entries[self.current_index]
            if "input" in entry:
                prompt = entry["input"].get("prompt", "")
                response = entry["input"].get("response", "")
            else:
                prompt = entry.get("prompt", "")
                response = entry.get("response", "")

            result = reflective_divergence_scan(prompt, response, m2_entry=entry)

            result_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "prompt": prompt,
                "response": response,
                **result
            }
            self.save_result(result_entry)

            self.textbox.delete(1.0, tk.END)
            self.textbox.insert(tk.END, json.dumps(result_entry, indent=2))

    def prev_entry(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.run_and_show()

    def next_entry(self):
        if self.current_index < len(self.entries) - 1:
            self.current_index += 1
            self.run_and_show()

    def save_result(self, entry):
        path = "M3_results.json"
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                try:
                    db = json.load(f)
                    if not isinstance(db, list):
                        db = [db]
                except json.JSONDecodeError:
                    db = []
        else:
            db = []

        db.append(entry)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(db, f, ensure_ascii=False, indent=2)

# === Run ===
if __name__ == "__main__":
    root = tk.Tk()
    app = RDE_GUI(root)
    root.mainloop()
