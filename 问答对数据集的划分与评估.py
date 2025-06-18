import tkinter as tk
from tkinter import scrolledtext, messagebox, ttk
import subprocess
import tempfile
import os
import sys
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import Ollama
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)

# Model configuration
MODEL_NAME = "qwen3:32b-q8_0"
API_BASE = "http://163.184.132.210:11434"

class PlaywrightCodeInterpreter:
    def __init__(self, root):
        self.root = root
        self.root.title("Playwright Code Interpreter")
        self.root.geometry("1000x850")
        self.root.configure(bg="#f0f0f0")
        
        # Create LLM chains
        self.code_to_text_chain = self._create_code_to_text_chain()
        self.text_to_code_chain = self._create_text_to_code_chain()
        
        self._setup_ui()
        
    def _create_code_to_text_chain(self):
        """Create LLM chain for code to natural language conversion"""
        llm = Ollama(model=MODEL_NAME, base_url=API_BASE)
        
        template = """/no_think You are a professional Playwright code interpreter. Translate Playwright scripts to concise natural language instructions.
- Infer operation intent from all steps
- Include element locators (text/content) for precision
- Output only operational descriptions, no markdown/notes
INPUT: Playwright code (JavaScript/Python)
OUTPUT: Step-by-step natural language instructions"""
        
        system_message_prompt = SystemMessagePromptTemplate.from_template(template)
        human_template = "{text}"
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
        
        chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
        return LLMChain(llm=llm, prompt=chat_prompt)
    
    def _create_text_to_code_chain(self):
        """Create LLM chain for natural language to code conversion"""
        llm = Ollama(model=MODEL_NAME, base_url=API_BASE)
        
        template = """/no_think You are a Playwright code generator. Convert natural language to valid JavaScript code.
- Output pure code with no explanations
- Include necessary imports (e.g., @playwright/test)
- Wrap in test function with page parameter
- Use data-test selectors when possible
INPUT: Step-by-step natural language instructions
OUTPUT: Playwright test code (JavaScript)"""
        
        system_message_prompt = SystemMessagePromptTemplate.from_template(template)
        human_template = "{text}"
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
        
        chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
        return LLMChain(llm=llm, prompt=chat_prompt)
    
    def _setup_ui(self):
        """Set up the user interface"""
        # Create title
        title_label = tk.Label(self.root, text="Playwright Code Interpreter", font=("Arial", 16, "bold"), bg="#f0f0f0")
        title_label.pack(pady=20)
        
        # Create main frame
        main_frame = tk.Frame(self.root, bg="#f0f0f0")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Code input frame
        code_input_frame = tk.LabelFrame(main_frame, text="Playwright Code Input", font=("Arial", 12), bg="#f0f0f0")
        code_input_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.code_input = scrolledtext.ScrolledText(code_input_frame, wrap=tk.WORD, width=80, height=10, font=("Consolas", 10))
        self.code_input.pack(fill=tk.BOTH, expand=True, pady=5, padx=5)
        
        # Natural language frame
        nl_frame = tk.LabelFrame(main_frame, text="Natural Language Description", font=("Arial", 12), bg="#f0f0f0")
        nl_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.natural_language = scrolledtext.ScrolledText(nl_frame, wrap=tk.WORD, width=80, height=8, font=("Arial", 10))
        self.natural_language.pack(fill=tk.BOTH, expand=True, pady=5, padx=5)
        
        # Code output frame
        code_output_frame = tk.LabelFrame(main_frame, text="Playwright Code Output", font=("Arial", 12), bg="#f0f0f0")
        code_output_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.code_output = scrolledtext.ScrolledText(code_output_frame, wrap=tk.WORD, width=80, height=10, font=("Consolas", 10))
        self.code_output.pack(fill=tk.BOTH, expand=True, pady=5, padx=5)
        
        # Button frame
        button_frame = tk.Frame(self.root, bg="#f0f0f0")
        button_frame.pack(fill=tk.X, pady=10)
        
        # Code to text button
        self.code_to_text_btn = tk.Button(button_frame, text="Code → Natural Language", command=self._convert_code_to_text,
                                         bg="#4CAF50", fg="white", font=("Arial", 10, "bold"))
        self.code_to_text_btn.pack(side=tk.LEFT, padx=5)
        
        # Text to code button
        self.text_to_code_btn = tk.Button(button_frame, text="Natural Language → Code", command=self._convert_text_to_code,
                                         bg="#2196F3", fg="white", font=("Arial", 10, "bold"))
        self.text_to_code_btn.pack(side=tk.LEFT, padx=5)
        
        # Execute code button
        self.execute_btn = tk.Button(button_frame, text="Execute Code", command=self._execute_code,
                                    bg="#f44336", fg="white", font=("Arial", 10, "bold"))
        self.execute_btn.pack(side=tk.LEFT, padx=5)
        
        # Load example button
        self.example_btn = tk.Button(button_frame, text="Load Example", command=self._load_example,
                                    bg="#FFC107", fg="black", font=("Arial", 10, "bold"))
        self.example_btn.pack(side=tk.LEFT, padx=5)
        
        # Clear all button
        self.clear_btn = tk.Button(button_frame, text="Clear All", command=self._clear_all,
                                  bg="#607D8B", fg="white", font=("Arial", 10, "bold"))
        self.clear_btn.pack(side=tk.LEFT, padx=5)
        
        # Execution result frame
        result_frame = tk.LabelFrame(self.root, text="Execution Result", font=("Arial", 12), bg="#f0f0f0")
        result_frame.pack(fill=tk.BOTH, expand=True, pady=10, padx=20)
        
        self.result_output = scrolledtext.ScrolledText(result_frame, wrap=tk.WORD, width=80, height=5, font=("Consolas", 10))
        self.result_output.pack(fill=tk.BOTH, expand=True, pady=5, padx=5)
        
        # Status bar
        self.status_bar = tk.Label(self.root, text="Ready", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def _convert_code_to_text(self):
        """Convert Playwright code to natural language"""
        code = self.code_input.get("1.0", tk.END).strip()
        if not code:
            messagebox.showwarning("Warning", "Please enter Playwright code")
            return
        
        try:
            self.status_bar.config(text="Converting code to natural language...")
            self.root.update()
            
            result = self.code_to_text_chain.run(text=code)
            
            self.natural_language.delete("1.0", tk.END)
            self.natural_language.insert(tk.END, result)
            self.status_bar.config(text="Conversion completed")
        except Exception as e:
            messagebox.showerror("Error", f"Error during conversion: {str(e)}")
            self.status_bar.config(text="Ready")
    
    def _convert_text_to_code(self):
        """Convert natural language description to Playwright code"""
        text = self.natural_language.get("1.0", tk.END).strip()
        if not text:
            messagebox.showwarning("Warning", "Please enter a natural language description")
            return
        
        try:
            self.status_bar.config(text="Generating Playwright code...")
            self.root.update()
            
            result = self.text_to_code_chain.run(text=text)
            
            # Cleanse output: remove leading/trailing markdown and empty lines
            cleaned_code = self._cleanse_code_output(result)
            
            self.code_output.delete("1.0", tk.END)
            self.code_output.insert(tk.END, cleaned_code)
            self.status_bar.config(text="Code generation completed")
        except Exception as e:
            messagebox.showerror("Error", f"Error generating code: {str(e)}")
            self.status_bar.config(text="Ready")
    
    def _cleanse_code_output(self, code):
        """Remove irrelevant context from LLM output"""
        # Remove common LLM response prefixes
        prefixes = ["```javascript", "```", "javascript", "// Code:", "// Output:"]
        for prefix in prefixes:
            if code.startswith(prefix):
                code = code[len(prefix):].lstrip()
        
        # Remove common suffixes
        if code.endswith("```"):
            code = code[:-3].rstrip()
        
        # Remove empty leading/trailing lines
        return "\n".join([line for line in code.split("\n") if line.strip()])
    
    def _execute_code(self):
        """Execute Playwright code with improved path handling"""
        code = self.code_output.get("1.0", tk.END).strip()
        if not code:
            messagebox.showwarning("Warning", "Please generate Playwright code first")
            return
        
        try:
            # Create temporary file in system temp directory
            with tempfile.NamedTemporaryFile(mode='w', suffix='.spec.js', delete=False, encoding='utf-8') as f:
                f.write(code)
                temp_file_path = f.name
            
            self.status_bar.config(text="Executing code...")
            self.root.update()
            
            # Clear output
            self.result_output.delete("1.0", tk.END)
            
            # Execute command with platform-specific handling
            self._run_command(temp_file_path)
            
            self.status_bar.config(text="Execution completed")
        except Exception as e:
            messagebox.showerror("Execution Error", f"Failed to execute code: {str(e)}")
            self.status_bar.config(text="Execution failed")
        finally:
            # Clean up temporary file
            if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
                os.remove(temp_file_path)
    
    def _run_command(self, file_path):
        """Run Playwright test command with platform compatibility"""
        # Determine command based on OS
        if sys.platform.startswith('win'):
            # Windows: try npx in default installation path
            commands = [
                ["npx", "playwright", "test", file_path, "--headed"],
                ["C:\\Program Files\\nodejs\\npx.cmd", "playwright", "test", file_path, "--headed"],
                ["C:\\Program Files (x86)\\nodejs\\npx.cmd", "playwright", "test", file_path, "--headed"]
            ]
        else:
            # macOS/Linux
            commands = [["npx", "playwright", "test", file_path, "--headed"]]
        
        # Try each command until one works
        last_error = None
        for cmd in commands:
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=120
                )
                
                if result.returncode == 0:
                    self.result_output.insert(tk.END, "Execution successful!\n\nStandard output:\n" + result.stdout)
                    return
                else:
                    last_error = f"Execution failed (return code: {result.returncode})\n\n" \
                               f"Standard output:\n{result.stdout}\n\n" \
                               f"Error output:\n{result.stderr}"
            except subprocess.TimeoutExpired:
                last_error = "Execution timed out (exceeded 120 seconds)"
            except FileNotFoundError:
                last_error = f"Command not found: {cmd[0]}. Please ensure Node.js and Playwright are installed."
            except Exception as e:
                last_error = f"Unexpected error: {str(e)}"
        
        # If all commands failed, display the last error
        self.result_output.insert(tk.END, last_error)
    
    def _load_example(self):
        """Load example data"""
        # Clear all fields
        self._clear_all()
        
        # Load code example
        example_code = """import { test, expect } from '@playwright/test';
test('test', async ({ page }) => {
    await page.goto('https://www.saucedemo.com/');
    await page.locator('[data-test="username"]').click();
    await page.locator('[data-test="username"]').fill('standard_user');
    await page.locator('[data-test="password"]').click();
    await page.locator('[data-test="password"]').fill('secret_sauce');
    await page.locator('[data-test="login-button"]').click();
    await page.locator('[data-test="item-1-title-link"]').click();
    await page.locator('[data-test="add-to-cart"]').click();
    await page.locator('[data-test="shopping-cart-link"]').click();
    await page.locator('[data-test="remove-sauce-labs-bolt-t-shirt"]').click();
    await page.locator('[data-test="continue-shopping"]').click();
    await page.getByRole('button', { name: 'Open Menu' }).click();
    await page.locator('[data-test="logout-sidebar-link"]').click();
});"""
        
        self.code_input.insert(tk.END, example_code)
        self.status_bar.config(text="Example data loaded")
    
    def _clear_all(self):
        """Clear all text areas"""
        self.code_input.delete("1.0", tk.END)
        self.natural_language.delete("1.0", tk.END)
        self.code_output.delete("1.0", tk.END)
        self.result_output.delete("1.0", tk.END)
        self.status_bar.config(text="Ready")

if __name__ == "__main__":
    root = tk.Tk()
    app = PlaywrightCodeInterpreter(root)
    root.mainloop()
