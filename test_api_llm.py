"""
Smart Testing Agent using OpenRouter LLM
Tests app.py comprehensively with full system knowledge and natural conversation flow
Tracks tool usage, routing, error handling, and functionality
"""

import json
import os
import re
import uuid
from typing import Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import requests
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Mistral Configuration (fallback from exhausted OpenRouter)
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MISTRAL_BASE_URL = "https://api.mistral.ai/v1"
LLM_MODEL = os.getenv("LLM_MODEL")

# App Configuration
APP_BASE_URL = "http://localhost:5000/"
#APP_BASE_URL = "https://mediflow-ai-medical-assistant-785629432566.us-central1.run.app/"
APP_CHAT_ENDPOINT = f"{APP_BASE_URL}/api/chat"
APP_HEALTH_ENDPOINT = f"{APP_BASE_URL}/api/health"

# Initialize Mistral Client
client = OpenAI(
    base_url=MISTRAL_BASE_URL,
    api_key=MISTRAL_API_KEY,
)


@dataclass
class TestMetrics:
    """Track metrics for each test turn"""
    turn_number: int
    user_message: str
    app_response: str
    expected_tools: list
    detected_tools: list
    expected_keywords: list
    found_keywords: list
    tools_match: bool
    keywords_match: bool
    response_quality: str
    timestamp: str


class PatientContextManager:
    """Manages patient database context for testing LLM"""

    def __init__(self, patient_file: str = "data/patients.json"):
        self.patients = self._load_patients(patient_file)
        self.system_knowledge = self._build_system_knowledge()

    def _load_patients(self, patient_file: str) -> list:
        """Load patient data from JSON"""
        if not os.path.exists(patient_file):
            print(f"Warning: {patient_file} not found")
            return []
        with open(patient_file, 'r') as f:
            return json.load(f)

    def _build_system_knowledge(self) -> str:
        """Build comprehensive system knowledge for testing LLM"""
        patients_summary = "\n".join(
            [f"- {p['patient_name']}: {p['primary_diagnosis']}" for p in self.patients[:5]]
        )
        
        return f"""
=== SYSTEM UNDER TEST: Multi-Agent Clinical Chatbot ===

PATIENT DATABASE (First 5):
{patients_summary}

SYSTEM ARCHITECTURE:
- Agent 1: Receptionist Agent - Greets patients, identifies them, routes to clinical agent
- Agent 2: Clinical Agent - Answers medical questions, provides guidance, uses tools

AVAILABLE TOOLS:
1. query_nephrology_docs (RAG Tool): Searches nephrology knowledge base for medical information
   - Triggers: Questions about kidney diseases, medical conditions, treatment options
   - Keywords: "disease", "treatment", "stages", "syndrome", "nephropathy"

2. patient_data_retrieval: Retrieves patient discharge information from database
   - Triggers: When patient is identified, asking about their medications, restrictions, follow-up
   - Keywords: "medication", "discharge", "restriction", "follow-up", "warning"

3. search_web (Online Search): Searches internet for medical information
   - Triggers: Questions outside knowledge base, current events, latest treatments
   - Keywords: "research", "latest", "update", "current", "new treatment"

EXPECTED BEHAVIORS:
- Receptionist greets patient and asks for name
- Once patient identified, clinical agent accesses their discharge info
- Medical questions should trigger RAG tool for knowledge
- Patient-specific questions should use data retrieval tool
- Complex queries may chain multiple tools

TESTING FOCUS:
- Tool invocation correctness
- Agent routing accuracy
- Context preservation across turns
- Error handling for invalid inputs
- Edge cases and boundary conditions
"""

    def get_context_for_llm(self) -> str:
        """Get formatted context for the testing LLM"""
        return self.system_knowledge + "\n\nGENERATE NATURAL CONVERSATION QUESTIONS TO TEST ALL ASPECTS OF THE SYSTEM"


class SmartTestingAgent:
    """Intelligent LLM-based testing agent"""

    def __init__(self, patient_context: PatientContextManager):
        self.patient_context = patient_context
        self.conversation_history = []
        self.test_metrics = []
        self.thread_id = f"test-thread-{str(uuid.uuid4())[:8]}"

    def generate_test_questions(self, test_scenario: str) -> list:
        """Use LLM to generate natural test questions for a scenario"""
        prompt = f"""
{self.patient_context.get_context_for_llm()}

TEST SCENARIO: {test_scenario}

Generate 3-4 natural questions a patient would ask in this scenario.
Return as a JSON array of strings.
Focus on testing different aspects of the system.

Return ONLY valid JSON array, no other text.
Example format: ["question 1", "question 2", "question 3"]
"""
        
        try:
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=500,
            )
            
            response_text = response.choices[0].message.content.strip()
            # Extract JSON array from response
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if json_match:
                questions = json.loads(json_match.group())
                return questions
            return []
        except Exception as e:
            print(f"Error generating test questions: {e}")
            return []

    def assess_response_quality(self, user_message: str, app_response: str, detected_tools: list) -> str:
        """Use LLM to assess response quality"""
        assessment_prompt = f"""
Assess this chatbot response quality in one word (excellent/good/adequate/poor):

User asked: "{user_message}"
Chatbot responded: "{app_response}"
Tools used: {detected_tools}

Criteria:
- Relevance to question
- Appropriate tool usage
- Clarity and helpfulness
- Medical accuracy (if applicable)

Return only one word assessment.
"""
        
        try:
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": assessment_prompt}],
                temperature=0.3,
                max_tokens=50,
            )
            return response.choices[0].message.content.strip().lower()
        except Exception as e:
            return "unknown"

    def detect_tools_in_response(self, response: str) -> list:
        """Detect which tools were used based on response content"""
        tools_detected = []

        # Keywords for patient_data_retrieval
        patient_keywords = [
            "medication", "discharge", "restriction", "follow-up",
            "warning", "instruction", "dosage", "daily", "twice daily"
        ]
        if any(keyword.lower() in response.lower() for keyword in patient_keywords):
            tools_detected.append("patient_data_retrieval")

        # Keywords for query_nephrology_docs (RAG)
        rag_keywords = [
            "kidney", "renal", "disease", "chronic", "acute",
            "glomerulonephritis", "nephropathy", "syndrome", "dialysis",
            "creatinine", "stage", "treatment", "management", "pathophysiology"
        ]
        if any(keyword.lower() in response.lower() for keyword in rag_keywords):
            tools_detected.append("query_nephrology_docs")

        # Keywords for search_web
        web_keywords = [
            "research", "study", "latest", "recent", "update",
            "current treatment", "new therapy", "published", "trial"
        ]
        if any(keyword.lower() in response.lower() for keyword in web_keywords):
            tools_detected.append("search_web")

        return tools_detected

    def send_message_to_app(self, message: str) -> str:
        """Send message to app.py and get response"""
        payload = {
            "message": message,
            "thread_id": self.thread_id
        }
        
        try:
            response = requests.post(
                APP_CHAT_ENDPOINT,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get("response", "No response")
            else:
                return f"Error: {response.status_code} - {response.text}"
        except requests.exceptions.RequestException as e:
            return f"Connection error: {str(e)}"

    def run_test_scenario(self, scenario_name: str, scenario_description: str, expected_tools: list) -> dict:
        """Run a complete test scenario"""
        print(f"\n{'='*80}")
        print(f"[TEST SCENARIO] {scenario_name}")
        print(f"{'='*80}")
        print(f"Description: {scenario_description}")
        print(f"Expected Tools: {expected_tools}\n")

        # Generate test questions
        questions = self.generate_test_questions(scenario_description)
        if not questions:
            print(f"Failed to generate questions for scenario")
            return {"scenario": scenario_name, "status": "FAILED", "metrics": []}

        scenario_metrics = []
        tools_used_overall = set()

        # Run conversation turns
        for turn_num, question in enumerate(questions, 1):
            print(f"\n[TURN {turn_num}] User: {question}")

            # Send to app
            app_response = self.send_message_to_app(question)
            print(f"[TURN {turn_num}] App: {app_response[:200]}..." if len(app_response) > 200 else f"[TURN {turn_num}] App: {app_response}")

            # Detect tools
            detected_tools = self.detect_tools_in_response(app_response)
            
            # Check tools match
            expected_set = set(expected_tools)
            detected_set = set(detected_tools)
            tools_match = len(detected_set & expected_set) > 0  # At least one expected tool found

            # Assess quality
            quality = self.assess_response_quality(question, app_response, detected_tools)

            # Record metrics
            metric = TestMetrics(
                turn_number=turn_num,
                user_message=question,
                app_response=app_response,
                expected_tools=expected_tools,
                detected_tools=detected_tools,
                expected_keywords=[],  # Not used in LLM scenario
                found_keywords=[],
                tools_match=tools_match,
                keywords_match=True,
                response_quality=quality,
                timestamp=datetime.now().isoformat()
            )
            scenario_metrics.append(metric)
            tools_used_overall.update(detected_tools)

            # Print metrics
            print(f"[*] Expected tools: {expected_tools}")
            print(f"[*] Tools detected: {detected_tools}")
            print(f"[*] Tools match: {tools_match}")
            print(f"[*] Response quality: {quality}")

        # Determine scenario result
        passed_turns = sum(1 for m in scenario_metrics if m.tools_match)
        total_turns = len(scenario_metrics)
        scenario_status = "PASS" if passed_turns == total_turns else "PARTIAL" if passed_turns > 0 else "FAIL"

        return {
            "scenario": scenario_name,
            "status": scenario_status,
            "metrics": scenario_metrics,
            "passed_turns": passed_turns,
            "total_turns": total_turns,
            "tools_used": list(tools_used_overall)
        }


class ComprehensiveTestRunner:
    """Orchestrates comprehensive testing"""

    def __init__(self):
        self.patient_context = PatientContextManager()
        self.test_results = []

    def check_app_health(self) -> bool:
        """Verify app is running"""
        try:
            response = requests.get(APP_HEALTH_ENDPOINT, timeout=5)
            return response.status_code == 200
        except:
            return False

    def run_all_tests(self):
        """Run comprehensive test suite"""
        print(f"\n{'='*80}")
        print(f"COMPREHENSIVE LLM-BASED TESTING SUITE")
        print(f"{'='*80}")
        print(f"App URL: {APP_BASE_URL}")
        print(f"Testing LLM: {LLM_MODEL}")
        print(f"Timestamp: {datetime.now().isoformat()}\n")

        # Check app health
        if not self.check_app_health():
            print("[ERROR] App is not running! Please start app.py first.")
            return

        print("[OK] App is healthy\n")

        # Define test scenarios
        scenarios = [
            {
                "name": "PATIENT IDENTIFICATION & DATA RETRIEVAL",
                "description": "Patient introduces themselves and asks about their discharge information",
                "expected_tools": ["patient_data_retrieval"]
            },
            {
                "name": "MEDICAL KNOWLEDGE QUESTIONS (RAG)",
                "description": "Patient asks general medical questions about kidney diseases and treatment",
                "expected_tools": ["query_nephrology_docs"]
            },
            {
                "name": "PATIENT-SPECIFIC MEDICAL GUIDANCE",
                "description": "Patient asks about their specific condition and what they should do",
                "expected_tools": ["patient_data_retrieval", "query_nephrology_docs"]
            },
            {
                "name": "COMPLEX CLINICAL QUESTIONS",
                "description": "Patient asks complex questions requiring both patient data and medical knowledge",
                "expected_tools": ["patient_data_retrieval", "query_nephrology_docs"]
            },
            {
                "name": "FOLLOW-UP & WARNINGS",
                "description": "Patient reports symptoms and asks about warning signs and when to seek help",
                "expected_tools": ["patient_data_retrieval", "query_nephrology_docs"]
            },
        ]

        # Run each scenario
        for scenario in scenarios:
            agent = SmartTestingAgent(self.patient_context)
            result = agent.run_test_scenario(
                scenario["name"],
                scenario["description"],
                scenario["expected_tools"]
            )
            self.test_results.append(result)

        # Print summary
        self.print_summary()

    def print_summary(self):
        """Print comprehensive test summary"""
        print("\n" + "="*80)
        print("TEST SUMMARY REPORT")
        print("="*80)

        passed_scenarios = sum(1 for r in self.test_results if r["status"] == "PASS")
        partial_scenarios = sum(1 for r in self.test_results if r["status"] == "PARTIAL")
        failed_scenarios = sum(1 for r in self.test_results if r["status"] == "FAIL")
        total_scenarios = len(self.test_results)

        print(f"\n[STATISTICS]")
        print(f"  Total Scenarios: {total_scenarios}")
        print(f"  Passed: {passed_scenarios}")
        print(f"  Partial: {partial_scenarios}")
        print(f"  Failed: {failed_scenarios}")
        print(f"  Success Rate: {(passed_scenarios + partial_scenarios) / total_scenarios * 100:.1f}%")

        print(f"\n[DETAILED RESULTS]")
        for result in self.test_results:
            status_symbol = "[OK]" if result["status"] == "PASS" else "[~]" if result["status"] == "PARTIAL" else "[FAIL]"
            print(f"\n{status_symbol} {result['scenario']}")
            print(f"    Status: {result['status']}")
            print(f"    Turns: {result['passed_turns']}/{result['total_turns']} passed")
            print(f"    Tools Used: {result['tools_used']}")

        print(f"\n[OVERALL TOOLS COVERAGE]")
        all_tools = set()
        for result in self.test_results:
            all_tools.update(result["tools_used"])
        
        expected_tools = {"patient_data_retrieval", "query_nephrology_docs", "search_web"}
        covered = all_tools & expected_tools
        print(f"  Expected: {sorted(expected_tools)}")
        print(f"  Covered: {sorted(covered)}")
        print(f"  Coverage: {len(covered)}/{len(expected_tools)} tools tested")

        if "search_web" not in all_tools:
            print(f"\n  [WARNING] search_web tool not triggered in any scenario")
            print(f"  [ACTION] Consider scenarios requiring internet search results")

        print(f"\n[RECOMMENDATIONS]")
        if failed_scenarios > 0:
            print(f"  - {failed_scenarios} scenario(s) failed - investigate agent routing")
        if partial_scenarios > partial_scenarios // 2:
            print(f"  - Many partial results - review tool invocation logic")
        if "search_web" not in all_tools:
            print(f"  - search_web tool never triggered - check clinical agent prompting")
        
        print(f"\n{'='*80}\n")


def main():
    """Main entry point"""
    runner = ComprehensiveTestRunner()
    runner.run_all_tests()


if __name__ == "__main__":
    main()
