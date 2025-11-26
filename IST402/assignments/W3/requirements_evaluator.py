#!/usr/bin/env python3
"""
Requirements Evaluator for Week 3 RAG Assignment

This script evaluates if the implementation in W3_RAG_Assignment.ipynb
matches all the requirements specified in W3_RAG_Assignment_Requirements.md
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

class RequirementsEvaluator:
    """Evaluates notebook implementation against assignment requirements"""
    
    def __init__(self, notebook_path: str, requirements_path: str):
        self.notebook_path = Path(notebook_path)
        self.requirements_path = Path(requirements_path)
        self.results = {
            "total_requirements": 0,
            "passed": 0,
            "failed": 0,
            "warnings": 0,
            "details": []
        }
    
    def load_notebook(self) -> Dict:
        """Load and parse the Jupyter notebook"""
        with open(self.notebook_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def extract_code_cells(self, notebook: Dict) -> List[str]:
        """Extract all code from code cells"""
        code_cells = []
        for cell in notebook.get('cells', []):
            if cell.get('cell_type') == 'code':
                source = cell.get('source', [])
                if isinstance(source, list):
                    code = ''.join(source)
                else:
                    code = source
                code_cells.append(code)
        return code_cells
    
    def extract_markdown_cells(self, notebook: Dict) -> List[str]:
        """Extract all text from markdown cells"""
        markdown_cells = []
        for cell in notebook.get('cells', []):
            if cell.get('cell_type') == 'markdown':
                source = cell.get('source', [])
                if isinstance(source, list):
                    text = ''.join(source)
                else:
                    text = source
                markdown_cells.append(text)
        return markdown_cells
    
    def check_task_1_system_prompt(self, code_cells: List[str], markdown_cells: List[str]) -> Tuple[bool, str]:
        """Check Task 1: System Prompt requirements"""
        self.results["total_requirements"] += 1
        
        # Check for business context definition
        has_business_context = any('BUSINESS_CONTEXT' in code or 'business context' in code.lower() 
                                   for code in code_cells)
        
        # Check for system prompt
        has_system_prompt = any('SYSTEM_PROMPT' in code or 'system prompt' in code.lower() 
                               for code in code_cells)
        
        # Check for Mistral model usage
        has_mistral = any('mistralai/Mistral-7B-Instruct' in code or 'Mistral-7B-Instruct' in code 
                         for code in code_cells)
        
        if has_business_context and has_system_prompt:
            self.results["passed"] += 1
            return True, "✅ Task 1: System prompt and business context defined"
        else:
            self.results["failed"] += 1
            missing = []
            if not has_business_context:
                missing.append("business context")
            if not has_system_prompt:
                missing.append("system prompt")
            return False, f"❌ Task 1: Missing {', '.join(missing)}"
    
    def check_task_2_qa_database(self, code_cells: List[str]) -> Tuple[bool, str]:
        """Check Task 2: Q&A Database requirements"""
        self.results["total_requirements"] += 1
        
        # Count Q&A pairs
        qa_count = 0
        has_comments = False
        
        for code in code_cells:
            # Look for Q&A pairs in various formats
            qa_patterns = [
                r'faq_data\s*=\s*\[',  # List assignment
                r'qa_pairs\s*=\s*\[',   # Alternative variable name
                r'\(["\'].*?["\'],\s*["\'].*?["\']\)',  # Tuple pattern
            ]
            
            for pattern in qa_patterns:
                if re.search(pattern, code, re.IGNORECASE):
                    # Count tuples in the list
                    tuples = re.findall(r'\(["\'].*?["\'],\s*["\'].*?["\']\)', code)
                    qa_count = max(qa_count, len(tuples))
            
            # Check for comments
            if '#' in code and ('Q&A' in code or 'question' in code.lower() or 'answer' in code.lower()):
                has_comments = True
        
        if qa_count >= 10:
            self.results["passed"] += 1
            return True, f"✅ Task 2: Found {qa_count} Q&A pairs (required: 10-15)"
        else:
            self.results["failed"] += 1
            return False, f"❌ Task 2: Found only {qa_count} Q&A pairs (required: 10-15)"
    
    def check_task_3_faiss(self, code_cells: List[str]) -> Tuple[bool, str]:
        """Check Task 3: FAISS implementation"""
        self.results["total_requirements"] += 1
        
        has_faiss = any('FAISS' in code or 'faiss' in code.lower() for code in code_cells)
        has_embeddings = any('HuggingFaceEmbeddings' in code or 'embeddings' in code.lower() 
                           for code in code_cells)
        has_documents = any('Document' in code or 'documents' in code.lower() 
                          for code in code_cells)
        
        if has_faiss and has_embeddings and has_documents:
            self.results["passed"] += 1
            return True, "✅ Task 3: FAISS vector database implemented"
        else:
            self.results["failed"] += 1
            missing = []
            if not has_faiss:
                missing.append("FAISS")
            if not has_embeddings:
                missing.append("embeddings")
            if not has_documents:
                missing.append("documents")
            return False, f"❌ Task 3: Missing {', '.join(missing)}"
    
    def check_task_4_test_questions(self, code_cells: List[str]) -> Tuple[bool, str]:
        """Check Task 4: Test questions"""
        self.results["total_requirements"] += 1
        
        has_answerable = any('answerable_questions' in code.lower() or 'answerable' in code.lower() 
                           for code in code_cells)
        has_unanswerable = any('unanswerable_questions' in code.lower() or 'unanswerable' in code.lower() 
                             for code in code_cells)
        
        # Count questions
        answerable_count = 0
        unanswerable_count = 0
        
        for code in code_cells:
            if 'answerable_questions' in code.lower():
                # Count list items
                matches = re.findall(r'["\'].*?["\']', code)
                answerable_count = max(answerable_count, len([m for m in matches if m.strip('"\'')]))
            if 'unanswerable_questions' in code.lower():
                matches = re.findall(r'["\'].*?["\']', code)
                unanswerable_count = max(unanswerable_count, len([m for m in matches if m.strip('"\'')]))
        
        if has_answerable and has_unanswerable and answerable_count >= 5 and unanswerable_count >= 5:
            self.results["passed"] += 1
            return True, f"✅ Task 4: Found {answerable_count} answerable and {unanswerable_count} unanswerable questions"
        else:
            self.results["failed"] += 1
            issues = []
            if not has_answerable:
                issues.append("answerable questions")
            if not has_unanswerable:
                issues.append("unanswerable questions")
            if answerable_count < 5:
                issues.append(f"only {answerable_count} answerable (need 5+)")
            if unanswerable_count < 5:
                issues.append(f"only {unanswerable_count} unanswerable (need 5+)")
            return False, f"❌ Task 4: Issues - {', '.join(issues)}"
    
    def check_task_5_testing(self, code_cells: List[str]) -> Tuple[bool, str]:
        """Check Task 5: Testing implementation"""
        self.results["total_requirements"] += 1
        
        has_rag_function = any('rag_qa_system' in code.lower() or 'rag' in code.lower() 
                              for code in code_cells)
        has_testing = any('test' in code.lower() and ('answerable' in code.lower() or 'unanswerable' in code.lower()) 
                        for code in code_cells)
        
        if has_rag_function and has_testing:
            self.results["passed"] += 1
            return True, "✅ Task 5: RAG testing implemented"
        else:
            self.results["failed"] += 1
            missing = []
            if not has_rag_function:
                missing.append("RAG function")
            if not has_testing:
                missing.append("testing code")
            return False, f"❌ Task 5: Missing {', '.join(missing)}"
    
    def check_task_6_model_comparison(self, code_cells: List[str]) -> Tuple[bool, str]:
        """Check Task 6: Model comparison"""
        self.results["total_requirements"] += 1
        
        # Check for required models
        required_models = [
            'consciousAI/question-answering-generative-t5-v1-base-s-q-c',
            'deepset/roberta-base-squad2',
            'google-bert/bert-large-cased-whole-word-masking-finetuned-squad',
            'gasolsun/DynamicRAG-8B'
        ]
        
        found_models = []
        all_code = ' '.join(code_cells)
        
        for model in required_models:
            if model in all_code:
                found_models.append(model.split('/')[-1])
        
        # Check for model comparison/ranking
        has_ranking = any('rank' in code.lower() or 'comparison' in code.lower() 
                         for code in code_cells)
        has_models_dict = any('models_to_test' in code.lower() or 'models' in code.lower() 
                            for code in code_cells)
        
        if len(found_models) >= 4 and has_ranking:
            self.results["passed"] += 1
            return True, f"✅ Task 6: Found {len(found_models)}/4 required models and ranking"
        else:
            self.results["failed"] += 1
            issues = []
            if len(found_models) < 4:
                issues.append(f"only {len(found_models)}/4 required models")
            if not has_ranking:
                issues.append("no ranking/comparison")
            return False, f"❌ Task 6: Issues - {', '.join(issues)}"
    
    def check_reflection(self, markdown_cells: List[str]) -> Tuple[bool, str]:
        """Check Reflection section"""
        self.results["total_requirements"] += 1
        
        all_markdown = ' '.join(markdown_cells).lower()
        
        has_strengths = 'strength' in all_markdown
        has_weaknesses = 'weakness' in all_markdown or 'limitation' in all_markdown
        has_applications = 'application' in all_markdown or 'real-world' in all_markdown
        has_learnings = 'learning' in all_markdown or 'learned' in all_markdown
        
        if has_strengths and has_weaknesses and has_applications and has_learnings:
            self.results["passed"] += 1
            return True, "✅ Reflection: All sections completed"
        else:
            self.results["warnings"] += 1
            missing = []
            if not has_strengths:
                missing.append("strengths")
            if not has_weaknesses:
                missing.append("weaknesses")
            if not has_applications:
                missing.append("applications")
            if not has_learnings:
                missing.append("learnings")
            return False, f"⚠️ Reflection: Missing {', '.join(missing)}"
    
    def check_required_packages(self, code_cells: List[str]) -> Tuple[bool, str]:
        """Check if required packages are installed"""
        self.results["total_requirements"] += 1
        
        required_packages = [
            'transformers',
            'torch',
            'sentence-transformers',
            'faiss-cpu',
            'langchain',
            'langchain-community'
        ]
        
        all_code = ' '.join(code_cells).lower()
        found_packages = [pkg for pkg in required_packages if pkg in all_code]
        
        if len(found_packages) >= len(required_packages) - 1:  # Allow 1 missing
            self.results["passed"] += 1
            return True, f"✅ Packages: Found {len(found_packages)}/{len(required_packages)} required packages"
        else:
            self.results["warnings"] += 1
            missing = [pkg for pkg in required_packages if pkg not in all_code]
            return False, f"⚠️ Packages: Missing {', '.join(missing)}"
    
    def evaluate(self) -> Dict:
        """Run all evaluation checks"""
        print("=" * 70)
        print("WEEK 3 RAG ASSIGNMENT - REQUIREMENTS EVALUATOR")
        print("=" * 70)
        print(f"\nEvaluating: {self.notebook_path.name}")
        print(f"Against: {self.requirements_path.name}\n")
        
        try:
            notebook = self.load_notebook()
            code_cells = self.extract_code_cells(notebook)
            markdown_cells = self.extract_markdown_cells(notebook)
            
            print("Running evaluation checks...\n")
            
            # Run all checks
            checks = [
                self.check_task_1_system_prompt(code_cells, markdown_cells),
                self.check_task_2_qa_database(code_cells),
                self.check_task_3_faiss(code_cells),
                self.check_task_4_test_questions(code_cells),
                self.check_task_5_testing(code_cells),
                self.check_task_6_model_comparison(code_cells),
                self.check_reflection(markdown_cells),
                self.check_required_packages(code_cells),
            ]
            
            for passed, message in checks:
                self.results["details"].append({
                    "status": "PASS" if passed else "FAIL",
                    "message": message
                })
                print(message)
            
            # Calculate score
            score = (self.results["passed"] / self.results["total_requirements"]) * 100
            
            print("\n" + "=" * 70)
            print("EVALUATION SUMMARY")
            print("=" * 70)
            print(f"Total Requirements: {self.results['total_requirements']}")
            print(f"✅ Passed: {self.results['passed']}")
            print(f"❌ Failed: {self.results['failed']}")
            print(f"⚠️  Warnings: {self.results['warnings']}")
            print(f"\n📊 Score: {score:.1f}%")
            
            if score >= 90:
                print("\n🎉 EXCELLENT! Implementation meets all requirements!")
            elif score >= 75:
                print("\n✅ GOOD! Minor issues to address.")
            elif score >= 60:
                print("\n⚠️  NEEDS WORK! Several requirements missing.")
            else:
                print("\n❌ INCOMPLETE! Major requirements missing.")
            
            print("=" * 70)
            
            self.results["score"] = score
            return self.results
            
        except FileNotFoundError as e:
            print(f"❌ Error: File not found - {e}")
            return self.results
        except json.JSONDecodeError as e:
            print(f"❌ Error: Invalid JSON in notebook - {e}")
            return self.results
        except Exception as e:
            print(f"❌ Error: {e}")
            return self.results


def main():
    """Main function to run the evaluator"""
    # Paths relative to script location
    script_dir = Path(__file__).parent
    notebook_path = script_dir / "W3_RAG_Assignment.ipynb"
    requirements_path = script_dir / "W3_RAG_Assignment_Requirements.md"
    
    evaluator = RequirementsEvaluator(str(notebook_path), str(requirements_path))
    results = evaluator.evaluate()
    
    # Save results to file
    results_file = script_dir / "evaluation_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📄 Detailed results saved to: {results_file}")
    
    return results


if __name__ == "__main__":
    main()

