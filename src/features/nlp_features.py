import os
import pandas as pd
from typing import Dict, Any, List, Optional
from src.models.nlp.contract_nlp import ContractNLPAnalyzer

class NLPContractFeaturesExtractor:
    """
    NLP Contract Feature Extraction Pipeline.
    Uses ContractNLPAnalyzer (LayoutLM + Legal-BERT + spaCy NER) to parse legal text
    and extract structured features for credit scoring models.
    """
    def __init__(self, model_name_or_path: str = "nlpae/legal-bert-base-uncased", spacy_model: str = "en_core_web_sm"):
        self.analyzer = ContractNLPAnalyzer(model_name_or_path=model_name_or_path, spacy_model=spacy_model)
        
    def extract_features_from_text(self, project_id: str, contract_text: str) -> Dict[str, Any]:
        """
        Parses a contract text block and extracts quantitative/categorical features for credit risk scoring.
        """
        report = self.analyzer.generate_risk_report(contract_text)
        entities = report.get("entities", {})
        clause_findings = report.get("clause_findings", [])
        
        # Calculate feature variables
        num_parties = len(entities.get("parties", []))
        num_dates = len(entities.get("dates", []))
        num_amounts = len(entities.get("amounts", []))
        
        # Binary flags for critical clauses being classified with high confidence (> 0.25)
        clause_flags = {
            "has_force_majeure": 0,
            "has_termination_clause": 0,
            "has_payment_terms": 0,
            "has_dispute_resolution": 0,
            "has_change_in_law": 0,
            "has_material_adverse_effect": 0,
            "has_indexation": 0,
            "has_liability_limit": 0,
            "has_environmental_compliance": 0,
            "has_delay_liquidated_damages": 0,
            "has_subcontractor_default": 0,
            "has_governing_law": 0
        }
        
        for finding in clause_findings:
            clause_name = finding["classified_clause"]
            conf = finding["confidence"]
            if conf > 0.25:
                flag_name = f"has_{clause_name}"
                if flag_name in clause_flags:
                    clause_flags[flag_name] = 1
                    
        # Overall risk score (0.0 to 1.0)
        contract_risk_score = report.get("overall_risk_score", 30.0) / 100.0
        
        features = {
            "project_id": project_id,
            "contract_risk_score": round(float(contract_risk_score), 4),
            "num_parties": num_parties,
            "num_dates": num_dates,
            "num_amounts": num_amounts,
        }
        features.update(clause_flags)
        
        return features

    def compute_all_nlp_features(self, project_contracts: List[Dict[str, str]]) -> pd.DataFrame:
        """
        Processes a list of dictionaries, each containing 'project_id' and 'contract_text',
        and returns a pandas DataFrame of extracted NLP features.
        """
        features_list = []
        for item in project_contracts:
            pid = item["project_id"]
            text = item["contract_text"]
            features = self.extract_features_from_text(pid, text)
            features["event_timestamp"] = pd.Timestamp.now()
            features_list.append(features)
            
        return pd.DataFrame(features_list)

FEAST_METADATA = {
    "entity": "project_id",
    "features": [
        "contract_risk_score",
        "num_parties",
        "num_dates",
        "num_amounts",
        "has_force_majeure",
        "has_termination_clause",
        "has_payment_terms",
        "has_dispute_resolution",
        "has_change_in_law",
        "has_material_adverse_effect",
        "has_indexation",
        "has_liability_limit",
        "has_environmental_compliance",
        "has_delay_liquidated_damages",
        "has_subcontractor_default",
        "has_governing_law"
    ]
}
