import tkinter as tk
from tkinter import scrolledtext, messagebox, ttk, simpledialog, filedialog
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
import threading
import json

# Default configuration
DEFAULT_CONFIG = {
    "MODEL_NAME": "qwen3:32b-q8_0",
    "API_BASE": "http://163.184.132.210:11434"
}

CONFIG_FILE = "config.json"

def load_config():
    """Load configuration file, return default if not exists"""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                cfg = json.load(f)
                # Fill missing fields with defaults
                return {**DEFAULT_CONFIG, **cfg}
        except Exception as e:
            print(f"Failed to load config: {e}")
    return DEFAULT_CONFIG.copy()

def save_config(config):
    """Save configuration to file"""
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4)
        return True
    except Exception as e:
        messagebox.showerror("Save Failed", f"Unable to save config file: {str(e)}")
        return False

class ModelConfigDialog(tk.Toplevel):
    """Model Configuration Dialog"""
    
    def __init__(self, parent, current_config, on_save):
        super().__init__(parent)
        self.title("Model Configuration")
        self.geometry("400x150")
        self.resizable(False, False)
        self.current_config = current_config
        self.on_save = on_save
        
        # Set dialog modality
        self.transient(parent)
        self.grab_set()
        
        self._setup_ui()
        
    def _setup_ui(self):
        # Create main frame
        main_frame = tk.Frame(self, padx=20, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Model Name
        tk.Label(main_frame, text="Model Name:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.model_name_var = tk.StringVar(value=self.current_config["MODEL_NAME"])
        tk.Entry(main_frame, textvariable=self.model_name_var, width=40).grid(row=0, column=1, pady=5)
        
        # API Base URL
        tk.Label(main_frame, text="API Base URL:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.api_base_var = tk.StringVar(value=self.current_config["API_BASE"])
        tk.Entry(main_frame, textvariable=self.api_base_var, width=40).grid(row=1, column=1, pady=5)
        
        # Button frame
        btn_frame = tk.Frame(self)
        btn_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # Save button
        tk.Button(btn_frame, text="Save Configuration", command=self._save_config, bg="#4CAF50", fg="white").pack(side=tk.LEFT, padx=5)
        
        # Reset button
        tk.Button(btn_frame, text="Reset Defaults", command=self._reset_defaults).pack(side=tk.LEFT, padx=5)
        
        # Cancel button
        tk.Button(btn_frame, text="Cancel", command=self.destroy).pack(side=tk.RIGHT, padx=5)
    
    def _reset_defaults(self):
        """Reset to default configuration"""
        self.model_name_var.set(DEFAULT_CONFIG["MODEL_NAME"])
        self.api_base_var.set(DEFAULT_CONFIG["API_BASE"])
    
    def _save_config(self):
        """Save configuration and close dialog"""
        new_config = {
            "MODEL_NAME": self.model_name_var.get(),
            "API_BASE": self.api_base_var.get()
        }
        
        if save_config(new_config):
            self.on_save(new_config)
            self.destroy()

class PlaywrightCodeInterpreter:
    def __init__(self, root):
        self.root = root
        self.root.title("Playwright Code Interpreter")
        self.root.geometry("1000x850")
        self.root.configure(bg="#f0f0f0")
        self.config = load_config()
        self._setup_ui()
        self._init_chains()
        
    def _init_chains(self):
        """Initialize LLM chains with current configuration"""
        self.code_to_text_chain = self._create_code_to_text_chain()
        self.text_to_code_chain = self._create_text_to_code_chain()
    
    def _create_code_to_text_chain(self):
        """Create LLM chain for code to text conversion"""
        llm = Ollama(
            model=self.config["MODEL_NAME"],
            base_url=self.config["API_BASE"]
        )
        
        template = """You are a professional Playwright code interpreter. Translate Playwright scripts to concise natural language instructions.
         - Infer operation intent from all steps
         - Include element locators (text/content) for precision
         - Output only operational descriptions, no markdown/notes
         INPUT: Playwright code (Python)
         OUTPUT: Step-by-step natural language instructions
         """
       
        system_message_prompt = SystemMessagePromptTemplate.from_template(template)
        human_template = "{text}"
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
       
        chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
        return LLMChain(llm=llm, prompt=chat_prompt)
   
    def _create_text_to_code_chain(self):
        """Create LLM chain for text to code conversion"""
        llm = Ollama(
            model=self.config["MODEL_NAME"],
            base_url=self.config["API_BASE"]
        )
        
        template = """You are a Playwright code generator. Convert natural language to valid Python code.
         - Output pure code with no explanations
         - Include necessary imports (e.g., from playwright.sync_api import sync_playwright)
         - Wrap in a run function with playwright parameter
         - Use data-test selectors when possible
         - The generated code must be executable
         INPUT: Step-by-step natural language instructions
         OUTPUT: Playwright test code (Python)
         """
       
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
       
        # First row button frame
        button_frame1 = tk.Frame(self.root, bg="#f0f0f0")
        button_frame1.pack(fill=tk.X, pady=5)
       
        # Code to text button
        self.code_to_text_btn = tk.Button(button_frame1, text="Code → Natural Language", command=self._convert_code_to_text,
                                         bg="#4CAF50", fg="white", font=("Arial", 10, "bold"))
        self.code_to_text_btn.pack(side=tk.LEFT, padx=5)
       
        # Text to code button
        self.text_to_code_btn = tk.Button(button_frame1, text="Natural Language → Code", command=self._convert_text_to_code,
                                         bg="#2196F3", fg="white", font=("Arial", 10, "bold"))
        self.text_to_code_btn.pack(side=tk.LEFT, padx=5)
       
        # Execute code button
        self.execute_btn = tk.Button(button_frame1, text="Execute Code", command=self._execute_code,
                                    bg="#f44336", fg="white", font=("Arial", 10, "bold"))
        self.execute_btn.pack(side=tk.LEFT, padx=5)
       
        # Second row button frame
        button_frame2 = tk.Frame(self.root, bg="#f0f0f0")
        button_frame2.pack(fill=tk.X, pady=5)
       
        # Load example button
        self.example_btn = tk.Button(button_frame2, text="Load Example", command=self._load_example,
                                    bg="#FFC107", fg="black", font=("Arial", 10, "bold"))
        self.example_btn.pack(side=tk.LEFT, padx=5)
       
        # Clear all button
        self.clear_btn = tk.Button(button_frame2, text="Clear All", command=self._clear_all,
                                  bg="#607D8B", fg="white", font=("Arial", 10, "bold"))
        self.clear_btn.pack(side=tk.LEFT, padx=5)
       
        # Model settings button
        self.model_settings_btn = tk.Button(button_frame2, text="Model Settings", command=self._open_model_settings,
                                          bg="#9C27B0", fg="white", font=("Arial", 10, "bold"))
        self.model_settings_btn.pack(side=tk.LEFT, padx=5)
       
        # Execution result frame
        result_frame = tk.LabelFrame(self.root, text="Execution Result", font=("Arial", 12), bg="#f0f0f0")
        result_frame.pack(fill=tk.BOTH, expand=True, pady=10, padx=20)
       
        self.result_output = scrolledtext.ScrolledText(result_frame, wrap=tk.WORD, width=80, height=5, font=("Consolas", 10))
        self.result_output.pack(fill=tk.BOTH, expand=True, pady=5, padx=5)
       
        # Status bar
        self.status_bar = tk.Label(self.root, text=f"Ready - Model: {self.config['MODEL_NAME']}", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def _open_model_settings(self):
        """Open model configuration dialog"""
        dialog = ModelConfigDialog(self.root, self.config, self._update_config)
    
    def _update_config(self, new_config):
        """Update configuration and reinitialize LLM chains"""
        self.config = new_config
        self._init_chains()
        self.status_bar.config(text=f"Configuration updated - Model: {self.config['MODEL_NAME']}")
        messagebox.showinfo("Success", "Model configuration updated. Changes will take effect on next conversion.")
    
    def _convert_code_to_text(self):
        """Convert Playwright code to natural language"""
        code = self.code_input.get("1.0", tk.END).strip()
        if not code:
            messagebox.showwarning("Warning", "Please enter Playwright code")
            return
        
        def task():
            try:
                self.status_bar.config(text="Converting code to natural language...")
                result = self.code_to_text_chain.run(text=code)
                self.natural_language.delete("1.0", tk.END)
                self.natural_language.insert(tk.END, result)
                self.status_bar.config(text="Conversion completed")
            except Exception as e:
                messagebox.showerror("Error", f"Conversion failed: {str(e)}")
                self.status_bar.config(text="Ready")
        
        threading.Thread(target=task).start()

    def _convert_text_to_code(self):
        """Convert natural language to Playwright code"""
        text = self.natural_language.get("1.0", tk.END).strip()
        if not text:
            messagebox.showwarning("Warning", "Please enter natural language description")
            return
        
        def task():
            try:
                self.status_bar.config(text="Generating Playwright code...")
                result = self.text_to_code_chain.run(text=text)
                cleaned_code = self._cleanse_code_output(result)
                self.code_output.delete("1.0", tk.END)
                self.code_output.insert(tk.END, cleaned_code)
                self.status_bar.config(text="Code generation completed")
            except Exception as e:
                messagebox.showerror("Error", f"Code generation failed: {str(e)}")
                self.status_bar.config(text="Ready")
        
        threading.Thread(target=task).start()

    def _execute_code(self):
        """Execute generated Playwright code"""
        code = self.code_output.get("1.0", tk.END).strip()
        if not code:
            messagebox.showwarning("Warning", "Please generate Playwright code first")
            return
        
        def task():
            temp_file_path = None
            try:
                with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
                    f.write(code)
                    temp_file_path = f.name
                
                self.status_bar.config(text="Executing code...")
                self.result_output.delete("1.0", tk.END)
                self._run_command(temp_file_path)
                self.status_bar.config(text="Execution completed")
            except Exception as e:
                messagebox.showerror("Execution Error", f"Execution failed: {str(e)}")
                self.status_bar.config(text="Execution failed")
            finally:
                if temp_file_path and os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
        
        threading.Thread(target=task).start()
   
    def _cleanse_code_output(self, code):
        """Remove irrelevant context from LLM output"""
        # Remove common LLM response prefixes
        prefixes = ["```python", "```", "python", "# Code:", "# Output:"]
        for prefix in prefixes:
            if code.startswith(prefix):
                code = code[len(prefix):].lstrip()
       
        # Remove common suffixes
        if code.endswith("```"):
            code = code[:-3].rstrip()
       
        # Remove empty leading/trailing lines
        return "\n".join([line for line in code.split("\n") if line.strip()])
   
    def _run_command(self, file_path):
        """Run Python Playwright test with proper environment"""
        # Determine Python command based on OS
        if sys.platform.startswith('win'):
            # Windows
            commands = [
                [sys.executable, file_path],  # Use current Python interpreter
                ["python", file_path],
                ["python3", file_path]
            ]
        else:
            # macOS/Linux
            commands = [
                [sys.executable, file_path],
                ["python3", file_path],
                ["python", file_path]
            ]
       
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
                last_error = f"Command not found: {cmd[0]}. Please ensure Python and Playwright are installed."
            except Exception as e:
                last_error = f"Unexpected error: {str(e)}"
       
        # If all commands failed, display the last error
        self.result_output.insert(tk.END, last_error)
   
    def _load_example(self):
        """Load example data"""
        # Clear all fields
        self._clear_all()
       
        # Load Python code example
        example_code = """from playwright.sync_api import sync_playwright

def run(playwright):
    browser = playwright.chromium.launch(headless=False)
    context = browser.new_context()
    page = context.new_page()
    page.goto("https://www.saucedemo.com/")
    page.locator('[data-test="username"]').fill("standard_user")
    page.locator('[data-test="password"]').fill("secret_sauce")
    page.locator('[data-test="login-button"]').click()
    page.locator('[data-test="add-to-cart-sauce-labs-backpack"]').click()
    page.locator('[data-test="shopping-cart-link"]').click()
    page.locator('[data-test="checkout"]').click()
    page.locator('[data-test="firstName"]').fill("John")
    page.locator('[data-test="lastName"]').fill("Doe")
    page.locator('[data-test="postalCode"]').fill("12345")
    page.locator('[data-test="continue"]').click()
    page.locator('[data-test="finish"]').click()
    browser.close()

with sync_playwright() as playwright:
    run(playwright)"""
       
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
