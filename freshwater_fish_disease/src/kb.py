import pandas as pd

class TreatmentKB:
    def __init__(self, treatments_csv="data/kb/treatments.csv", refs_csv="data/kb/references.csv"):
        self.tr = pd.read_csv(treatments_csv)
        self.refs = pd.read_csv(refs_csv)
        self.refmap = dict(zip(self.refs["evidence_ref"], self.refs["citation"]))

    def treatments_for(self, disease_name_or_id):
        df = self.tr.copy()
        if disease_name_or_id.upper().startswith("D"):
            df = df[df["disease_id"]==disease_name_or_id]
        else:
            df = df[df["disease_name"].str.lower()==disease_name_or_id.lower()]
        df = df.assign(citation=df.get("evidence_ref","").map(self.refmap))
        cols = ["treatment_name","active_substance","dosage_guidance","cautions","source_link","citation"]
        return df[["disease_id","disease_name","synonyms"]+cols].reset_index(drop=True)
