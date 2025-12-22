# global_config.py
from typing import Optional
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

class BrainHeartSettings:
    def __init__(self, brain_provider: Optional[str], brain_model: Optional[str],
                 heart_provider: Optional[str], heart_model: Optional[str],
                 indic_provider: Optional[str], indic_model: Optional[str],
                 routing_provider: Optional[str], routing_model: Optional[str],
                 simple_whatsapp_provider: Optional[str], simple_whatsapp_model: Optional[str],
                 cot_whatsapp_provider: Optional[str], cot_whatsapp_model: Optional[str],
                 grievance_provider: Optional[str], grievance_model: Optional[str],
                 query_agent_provider: Optional[str], query_agent_model: Optional[str],
                 use_premium_search: bool, web_model: Optional[str],
                 grievance_agent_enabled: bool,
                 grievance_agent_provider: Optional[str], grievance_agent_model: Optional[str]):
        self.brain_provider = brain_provider
        self.brain_model = brain_model
        self.heart_provider = heart_provider
        self.heart_model = heart_model
        self.indic_provider = indic_provider
        self.indic_model = indic_model
        self.routing_provider = routing_provider
        self.routing_model = routing_model
        self.simple_whatsapp_provider = simple_whatsapp_provider
        self.simple_whatsapp_model = simple_whatsapp_model
        self.cot_whatsapp_provider = cot_whatsapp_provider
        self.cot_whatsapp_model = cot_whatsapp_model
        self.grievance_provider = grievance_provider
        self.grievance_model = grievance_model
        self.query_agent_provider = query_agent_provider
        self.query_agent_model = query_agent_model
        self.use_premium_search = use_premium_search
        self.web_model = web_model
        # GrievanceAgent toggle
        self.grievance_agent_enabled = grievance_agent_enabled
        self.grievance_agent_provider = grievance_agent_provider
        self.grievance_agent_model = grievance_agent_model

settings = BrainHeartSettings(
    brain_provider=os.getenv('BRAIN_LLM_PROVIDER'),
    brain_model=os.getenv('BRAIN_LLM_MODEL'),
    heart_provider=os.getenv('HEART_LLM_PROVIDER'),
    heart_model=os.getenv('HEART_LLM_MODEL'),
    indic_provider=os.getenv('INDIC_HEART_LLM_PROVIDER'),
    indic_model=os.getenv('INDIC_HEART_LLM_MODEL'),
    routing_provider=os.getenv('ROUTING_LLM_PROVIDER'),
    routing_model=os.getenv('ROUTING_LLM_MODEL'),
    simple_whatsapp_provider=os.getenv('SIMPLE_WHATSAPP_LLM_PROVIDER'),
    simple_whatsapp_model=os.getenv('SIMPLE_WHATSAPP_LLM_MODEL'),
    cot_whatsapp_provider=os.getenv('COT_WHATSAPP_LLM_PROVIDER'),
    cot_whatsapp_model=os.getenv('COT_WHATSAPP_LLM_MODEL'),
    grievance_provider=os.getenv('GRIEVANCE_LLM_PROVIDER'),
    grievance_model=os.getenv('GRIEVANCE_LLM_MODEL'),
    query_agent_provider=os.getenv('QUERY_AGENT_LLM_PROVIDER'),
    query_agent_model=os.getenv('QUERY_AGENT_LLM_MODEL'),
    use_premium_search=os.getenv('USE_PREMIUM_SEARCH', 'false').lower() == 'true',
    web_model=os.getenv('WEB_MODEL', None),
    # GrievanceAgent settings
    grievance_agent_enabled=os.getenv('GRIEVANCE_AGENT_ENABLED', 'false').lower() == 'true',
    grievance_agent_provider=os.getenv('GRIEVANCE_AGENT_PROVIDER', 'openrouter'),
    grievance_agent_model=os.getenv('GRIEVANCE_AGENT_MODEL', 'meta-llama/llama-3.3-70b-instruct')
)
