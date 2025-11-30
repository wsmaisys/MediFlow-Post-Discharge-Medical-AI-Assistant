"""
Dynamic Testing Agent with Code Analysis
Automatically analyzes codebase, generates test plans, executes tests, and provides recommendations
"""

import json
import os
import re
import uuid
from typing import Optional, List, Dict
from dataclasses import dataclass, asdict
from datetime import datetime
import requests
from dotenv import load_dotenv
from openai import OpenAI
from pathlib import Path

# Load environment variables
load_dotenv()

# Mistral Configuration
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MISTRAL_BASE_URL = "https://api.mistral.ai/v1"
LLM_MODEL = os.getenv("LLM_MODEL")

# App Configuration
APP_BASE_URL = "https://mediflow-ai-medical-assistant-785629432566.us-central1.run.app/"
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
    detected_tools: list
    response_quality: str
    issues_found: list
    timestamp: str


@dataclass
class TestScenario:
    """Dynamically generated test scenario"""
    scenario_id: str
    name: str
    description: str
    test_questions: List[str]
    expected_behaviors: List[str]
    expected_tools: List[str]


class CodebaseAnalyzer:
    """Analyzes the entire codebase to understand system architecture"""

    def __init__(self, root_folder: str = "."):
        self.root_folder = Path(root_folder)
        self.codebase_content = {}
        self.analysis_summary = ""

    def scan_codebase(self) -> Dict[str, str]:
        """Scan and read all Python and JSON files"""
        print(f"\n[SCANNING] Reading codebase from: {self.root_folder.absolute()}")
        
        file_types = ["*.py", "*.json"]
        files_read = 0
        
        for pattern in file_types:
            for file_path in self.root_folder.rglob(pattern):
                # Skip virtual environments, cache, and test files
                if any(skip in str(file_path) for skip in ["venv", "__pycache__", ".git", "node_modules"]):
                    continue
                
                try:
                    relative_path = file_path.relative_to(self.root_folder)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        self.codebase_content[str(relative_path)] = content
                        files_read += 1
                        print(f"  ✓ Read: {relative_path}")
                except Exception as e:
                    print(f"  ✗ Failed to read {file_path}: {e}")
        
        print(f"\n[OK] Successfully read {files_read} files\n")
        return self.codebase_content

    def generate_codebase_summary(self) -> str:
        """Generate a structured summary of the codebase for LLM analysis"""
        summary_parts = []
        
        for file_path, content in self.codebase_content.items():
            # Truncate very long files
            truncated_content = content[:3000] + "\n...[truncated]" if len(content) > 3000 else content
            summary_parts.append(f"\n{'='*60}\nFILE: {file_path}\n{'='*60}\n{truncated_content}")
        
        return "\n".join(summary_parts)

    def analyze_with_llm(self) -> str:
        """Use LLM to analyze the codebase and understand architecture"""
        codebase_summary = self.generate_codebase_summary()
        
        analysis_prompt = f"""
You are a software testing expert analyzing a codebase to understand its architecture.

CODEBASE CONTENTS:
{codebase_summary}

Analyze this codebase and provide a structured summary including:

1. SYSTEM ARCHITECTURE:
   - What type of application is this?
   - What are the main components/modules?
   - What agents/services are defined?

2. AVAILABLE TOOLS/FUNCTIONS:
   - List all tools/functions the system can invoke
   - Describe what each tool does
   - Identify trigger conditions for each tool

3. DATA MODELS:
   - What data structures are used?
   - What JSON files contain what data?
   - What is the expected data flow?

4. API ENDPOINTS:
   - What endpoints are available?
   - What are the request/response formats?

5. CONVERSATION FLOW:
   - How does the system handle user interactions?
   - What is the expected conversation pattern?

6. POTENTIAL TEST AREAS:
   - What aspects of the system should be tested?
   - What are potential edge cases?
   - What could go wrong?

Provide a comprehensive but concise analysis.
"""
        
        try:
            print("[ANALYZING] Using LLM to analyze codebase architecture...")
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": analysis_prompt}],
                temperature=0.3,
                max_tokens=2000,
            )
            
            self.analysis_summary = response.choices[0].message.content.strip()
            print("[OK] Codebase analysis complete\n")
            return self.analysis_summary
        
        except Exception as e:
            print(f"[ERROR] Failed to analyze codebase: {e}")
            return ""


class DynamicTestPlanGenerator:
    """Generates dynamic test plans based on code analysis"""

    def __init__(self, codebase_analysis: str):
        self.codebase_analysis = codebase_analysis
        self.test_plan = []

    def generate_test_plan(self) -> List[TestScenario]:
        """Use LLM to generate a comprehensive test plan"""
        
        test_plan_prompt = f"""
Based on this codebase analysis:

{self.codebase_analysis}

Generate a comprehensive test plan with 5-7 test scenarios.

For each scenario, provide:
1. scenario_id: unique identifier (e.g., "TEST_001")
2. name: descriptive name
3. description: what this scenario tests
4. test_questions: 3-4 natural user questions to test this scenario
5. expected_behaviors: what the system should do
6. expected_tools: which tools/functions should be invoked

Return ONLY a valid JSON array of test scenarios.

Example format:
[
  {{
    "scenario_id": "TEST_001",
    "name": "Patient Identification Flow",
    "description": "Tests if the system correctly identifies patients and retrieves their data",
    "test_questions": [
      "Hi, my name is John Smith",
      "Can you tell me about my medications?",
      "What are my discharge instructions?"
    ],
    "expected_behaviors": [
      "Greets the patient",
      "Retrieves patient data from database",
      "Provides medication information"
    ],
    "expected_tools": ["patient_data_retrieval"]
  }}
]

Generate scenarios that test:
- Normal operation flows
- Tool invocation correctness
- Agent routing
- Error handling
- Edge cases
- Data retrieval accuracy
- Multi-turn conversations

Return ONLY valid JSON, no other text.
"""
        
        try:
            print("[GENERATING] Creating dynamic test plan...")
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": test_plan_prompt}],
                temperature=0.7,
                max_tokens=3000,
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Extract JSON array
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if json_match:
                test_scenarios_data = json.loads(json_match.group())
                
                # Convert to TestScenario objects
                self.test_plan = [
                    TestScenario(
                        scenario_id=s["scenario_id"],
                        name=s["name"],
                        description=s["description"],
                        test_questions=s["test_questions"],
                        expected_behaviors=s["expected_behaviors"],
                        expected_tools=s["expected_tools"]
                    )
                    for s in test_scenarios_data
                ]
                
                print(f"[OK] Generated {len(self.test_plan)} test scenarios\n")
                return self.test_plan
            else:
                print("[ERROR] Could not extract JSON from LLM response")
                return []
        
        except Exception as e:
            print(f"[ERROR] Failed to generate test plan: {e}")
            return []

    def display_test_plan(self):
        """Display the generated test plan"""
        print("="*80)
        print("DYNAMIC TEST PLAN")
        print("="*80)
        
        for i, scenario in enumerate(self.test_plan, 1):
            print(f"\n[{scenario.scenario_id}] {scenario.name}")
            print(f"  Description: {scenario.description}")
            print(f"  Expected Tools: {', '.join(scenario.expected_tools)}")
            print(f"  Test Questions: {len(scenario.test_questions)}")
            for j, question in enumerate(scenario.test_questions, 1):
                print(f"    {j}. {question}")
        
        print("\n" + "="*80 + "\n")


class DynamicTestExecutor:
    """Executes dynamic test scenarios"""

    def __init__(self):
        self.thread_id = f"test-thread-{str(uuid.uuid4())[:8]}"
        self.test_results = []

    def detect_tools_in_response(self, response: str) -> list:
        """Detect which tools were used based on response content"""
        tools_detected = []

        # Patient data retrieval keywords
        patient_keywords = [
            "medication", "discharge", "restriction", "follow-up",
            "warning", "instruction", "dosage", "daily", "twice daily"
        ]
        if any(keyword.lower() in response.lower() for keyword in patient_keywords):
            tools_detected.append("patient_data_retrieval")

        # RAG tool keywords
        rag_keywords = [
            "kidney", "renal", "disease", "chronic", "acute",
            "glomerulonephritis", "nephropathy", "syndrome", "dialysis",
            "creatinine", "stage", "treatment", "management", "pathophysiology"
        ]
        if any(keyword.lower() in response.lower() for keyword in rag_keywords):
            tools_detected.append("query_nephrology_docs")

        # Web search keywords
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

    def assess_response_with_llm(self, scenario: TestScenario, user_message: str, 
                                 app_response: str, detected_tools: list) -> Dict:
        """Use LLM to assess response quality and detect issues"""
        
        assessment_prompt = f"""
Assess this chatbot response:

SCENARIO: {scenario.name}
EXPECTED BEHAVIORS: {', '.join(scenario.expected_behaviors)}
EXPECTED TOOLS: {', '.join(scenario.expected_tools)}

USER ASKED: "{user_message}"
CHATBOT RESPONDED: "{app_response}"
DETECTED TOOLS: {detected_tools}

Provide assessment as JSON:
{{
  "quality": "excellent|good|adequate|poor",
  "issues_found": ["issue 1", "issue 2"],
  "tools_correct": true/false,
  "behavior_correct": true/false
}}

Return ONLY valid JSON.
"""
        
        try:
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": assessment_prompt}],
                temperature=0.3,
                max_tokens=500,
            )
            
            response_text = response.choices[0].message.content.strip()
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            return {
                "quality": "unknown",
                "issues_found": [],
                "tools_correct": False,
                "behavior_correct": False
            }
        except Exception as e:
            return {
                "quality": "error",
                "issues_found": [str(e)],
                "tools_correct": False,
                "behavior_correct": False
            }

    def execute_scenario(self, scenario: TestScenario) -> Dict:
        """Execute a single test scenario"""
        print(f"\n{'='*80}")
        print(f"[EXECUTING] {scenario.scenario_id}: {scenario.name}")
        print(f"{'='*80}")
        print(f"Description: {scenario.description}")
        print(f"Expected Tools: {scenario.expected_tools}\n")

        scenario_metrics = []
        all_issues = []

        for turn_num, question in enumerate(scenario.test_questions, 1):
            print(f"\n[TURN {turn_num}] User: {question}")

            # Send to app
            app_response = self.send_message_to_app(question)
            print(f"[TURN {turn_num}] App: {app_response[:200]}..." if len(app_response) > 200 else f"[TURN {turn_num}] App: {app_response}")

            # Detect tools
            detected_tools = self.detect_tools_in_response(app_response)

            # Assess with LLM
            assessment = self.assess_response_with_llm(scenario, question, app_response, detected_tools)

            # Record metrics
            metric = TestMetrics(
                turn_number=turn_num,
                user_message=question,
                app_response=app_response,
                detected_tools=detected_tools,
                response_quality=assessment["quality"],
                issues_found=assessment["issues_found"],
                timestamp=datetime.now().isoformat()
            )
            scenario_metrics.append(metric)
            all_issues.extend(assessment["issues_found"])

            # Print assessment
            print(f"[*] Expected Tools: {scenario.expected_tools}")
            print(f"[*] Detected Tools: {detected_tools}")
            print(f"[*] Quality: {assessment['quality']}")
            print(f"[*] Tools Correct: {assessment['tools_correct']}")
            print(f"[*] Behavior Correct: {assessment['behavior_correct']}")
            if assessment["issues_found"]:
                print(f"[!] Issues: {', '.join(assessment['issues_found'])}")

        # Determine scenario result
        quality_scores = {"excellent": 4, "good": 3, "adequate": 2, "poor": 1, "unknown": 0, "error": 0}
        avg_quality = sum(quality_scores.get(m.response_quality, 0) for m in scenario_metrics) / len(scenario_metrics)
        
        if avg_quality >= 3:
            status = "PASS"
        elif avg_quality >= 2:
            status = "PARTIAL"
        else:
            status = "FAIL"

        return {
            "scenario": scenario.name,
            "scenario_id": scenario.scenario_id,
            "status": status,
            "metrics": scenario_metrics,
            "issues": list(set(all_issues)),
            "average_quality": avg_quality
        }

    def execute_test_plan(self, test_plan: List[TestScenario]) -> List[Dict]:
        """Execute all test scenarios"""
        print("\n" + "="*80)
        print("EXECUTING DYNAMIC TEST PLAN")
        print("="*80 + "\n")

        for scenario in test_plan:
            result = self.execute_scenario(scenario)
            self.test_results.append(result)

        return self.test_results


class RecommendationEngine:
    """Generates recommendations based on test results and code analysis"""

    def __init__(self, codebase_analysis: str, test_results: List[Dict], codebase_content: Dict[str, str]):
        self.codebase_analysis = codebase_analysis
        self.test_results = test_results
        self.codebase_content = codebase_content

    def generate_recommendations(self):
        """Use LLM to generate comprehensive recommendations"""
        
        # Prepare test results summary
        test_summary = "\n".join([
            f"- {r['scenario_id']}: {r['status']} (Quality: {r['average_quality']:.2f}, Issues: {len(r['issues'])})"
            for r in self.test_results
        ])
        
        all_issues = []
        for r in self.test_results:
            all_issues.extend(r['issues'])
        
        recommendation_prompt = f"""
You are a senior software architect reviewing a chatbot system.

CODEBASE ANALYSIS:
{self.codebase_analysis}

TEST RESULTS:
{test_summary}

ISSUES FOUND:
{chr(10).join(set(all_issues)) if all_issues else "No specific issues found"}

Based on the code analysis and test results, provide comprehensive recommendations in these categories:

1. CRITICAL ISSUES
   - Bugs or problems that must be fixed immediately
   - Security concerns
   - Data integrity issues

2. ARCHITECTURE IMPROVEMENTS
   - Code structure improvements
   - Design pattern suggestions
   - Scalability concerns

3. TOOL & AGENT IMPROVEMENTS
   - Tool invocation accuracy
   - Agent routing optimization
   - Context handling

4. ERROR HANDLING
   - Missing error cases
   - Graceful degradation
   - User experience during errors

5. TESTING IMPROVEMENTS
   - Test coverage gaps
   - Additional test scenarios needed
   - Edge cases not covered

6. PERFORMANCE OPTIMIZATION
   - Response time improvements
   - Resource usage
   - Caching opportunities

7. USER EXPERIENCE
   - Conversation flow improvements
   - Response clarity
   - Helpful error messages

8. CODE QUALITY
   - Code organization
   - Documentation
   - Maintainability

For each recommendation:
- Be specific and actionable
- Reference actual code locations if possible
- Explain WHY the change is needed
- Suggest HOW to implement it

Provide detailed, professional recommendations in Markdown format.
"""
        
        try:
            print("\n" + "="*80)
            print("GENERATING RECOMMENDATIONS")
            print("="*80 + "\n")
            print("[ANALYZING] Generating recommendations based on tests and code review...\n")
            
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": recommendation_prompt}],
                temperature=0.5,
                max_tokens=3000,
            )
            
            recommendations = response.choices[0].message.content.strip()
            
            # Display in terminal
            print("="*80)
            print("RECOMMENDATIONS REPORT")
            print("="*80)
            print(recommendations)
            print("\n" + "="*80 + "\n")
            
            # Save to recommendations.md file
            self.save_recommendations_to_file(recommendations, test_summary, all_issues)
            
            return recommendations
        
        except Exception as e:
            print(f"[ERROR] Failed to generate recommendations: {e}")
            return ""

    def save_recommendations_to_file(self, recommendations: str, test_summary: str, all_issues: list):
        """Save recommendations to recommendations.md file"""
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            markdown_content = f"""# Testing Recommendations Report

**Generated:** {timestamp}  
**Testing LLM:** {LLM_MODEL}

---

## Test Execution Summary

{test_summary}

---

## Issues Identified

{chr(10).join(f"- {issue}" for issue in set(all_issues)) if all_issues else "No specific issues found"}

---

## Detailed Recommendations

{recommendations}

---

*This report was automatically generated by the Dynamic Testing Agent*
"""
            
            output_file = Path("recommendations.md")
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            
            print(f"[OK] Recommendations saved to: {output_file.absolute()}\n")
            
        except Exception as e:
            print(f"[ERROR] Failed to save recommendations to file: {e}\n")


class AutomatedTestingPipeline:
    """Orchestrates the entire automated testing pipeline"""

    def __init__(self, root_folder: str = "."):
        self.root_folder = root_folder
        self.codebase_analyzer = CodebaseAnalyzer(root_folder)
        self.test_executor = DynamicTestExecutor()

    def check_app_health(self) -> bool:
        """Verify app is running"""
        try:
            response = requests.get(APP_HEALTH_ENDPOINT, timeout=5)
            return response.status_code == 200
        except:
            return False

    def run_pipeline(self):
        """Run the complete automated testing pipeline"""
        print("\n" + "="*80)
        print("AUTOMATED DYNAMIC TESTING PIPELINE")
        print("="*80)
        print(f"Root Folder: {Path(self.root_folder).absolute()}")
        print(f"Testing LLM: {LLM_MODEL}")
        print(f"Timestamp: {datetime.now().isoformat()}")
        print("="*80 + "\n")

        # Step 1: Check app health
        print("[STEP 1] Checking application health...")
        if not self.check_app_health():
            print("[ERROR] App is not running! Please start app.py first.")
            return
        print("[OK] Application is healthy\n")

        # Step 2: Scan and analyze codebase
        print("[STEP 2] Scanning and analyzing codebase...")
        self.codebase_analyzer.scan_codebase()
        codebase_analysis = self.codebase_analyzer.analyze_with_llm()
        
        if not codebase_analysis:
            print("[ERROR] Failed to analyze codebase")
            return

        print("\n" + "="*80)
        print("CODEBASE ANALYSIS")
        print("="*80)
        print(codebase_analysis)
        print("="*80 + "\n")

        # Step 3: Generate dynamic test plan
        print("[STEP 3] Generating dynamic test plan...")
        test_plan_generator = DynamicTestPlanGenerator(codebase_analysis)
        test_plan = test_plan_generator.generate_test_plan()
        
        if not test_plan:
            print("[ERROR] Failed to generate test plan")
            return
        
        test_plan_generator.display_test_plan()

        # Step 4: Execute test plan
        print("[STEP 4] Executing dynamic test plan...")
        test_results = self.test_executor.execute_test_plan(test_plan)

        # Step 5: Print test summary
        self.print_test_summary(test_results)

        # Step 6: Generate recommendations
        print("[STEP 5] Generating recommendations based on code review and test results...")
        recommendation_engine = RecommendationEngine(
            codebase_analysis,
            test_results,
            self.codebase_analyzer.codebase_content
        )
        recommendation_engine.generate_recommendations()

    def print_test_summary(self, test_results: List[Dict]):
        """Print test execution summary"""
        print("\n" + "="*80)
        print("TEST EXECUTION SUMMARY")
        print("="*80)

        passed = sum(1 for r in test_results if r["status"] == "PASS")
        partial = sum(1 for r in test_results if r["status"] == "PARTIAL")
        failed = sum(1 for r in test_results if r["status"] == "FAIL")
        total = len(test_results)

        print(f"\nTotal Scenarios: {total}")
        print(f"Passed: {passed}")
        print(f"Partial: {partial}")
        print(f"Failed: {failed}")
        print(f"Success Rate: {(passed / total * 100):.1f}%")

        print("\nDetailed Results:")
        for result in test_results:
            status_symbol = "✓" if result["status"] == "PASS" else "~" if result["status"] == "PARTIAL" else "✗"
            print(f"\n{status_symbol} [{result['scenario_id']}] {result['scenario']}")
            print(f"  Status: {result['status']}")
            print(f"  Average Quality: {result['average_quality']:.2f}/4.0")
            if result['issues']:
                print(f"  Issues: {', '.join(result['issues'][:3])}{'...' if len(result['issues']) > 3 else ''}")

        print("\n" + "="*80 + "\n")


def main():
    """Main entry point"""
    pipeline = AutomatedTestingPipeline(root_folder=".")
    pipeline.run_pipeline()


if __name__ == "__main__":
    main()