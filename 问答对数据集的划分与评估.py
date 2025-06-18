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
        self.root.geometry("1200x900")
        
        # Configuration options
        self.config = {
            "headless": tk.BooleanVar(value=False),
            "model_name": tk.StringVar(value=MODEL_NAME),
            "api_base": tk.StringVar(value=API_BASE),
            "timeout": tk.IntVar(value=60)
        }
        
        # Set theme
        self.style = ttk.Style()
        if 'clam' in self.style.theme_names():
            self.style.theme_use('clam')
        
        # Create LLM chains
        self.code_to_text_chain = self._create_code_to_text_chain()
        self.text_to_code_chain = self._create_text_to_code_chain()
        
        self._setup_ui()
        self._setup_config_panel()
        
    def _create_code_to_text_chain(self):
        """Create LLM chain for code to natural language conversion"""
        llm = Ollama(
            model=self.config["model_name"].get(),
            base_url=self.config["api_base"].get(),
            timeout=self.config["timeout"].get()
        )
        
        template = """
        You are a professional Playwright code interpreter. Convert Python Playwright code into clear, concise natural language instructions.
        
        Output format:
        1. Step-by-step description of actions
        2. Include element locators (e.g., "[data-test='login-button']")
        3. Do not include code or markdown syntax
        
        Example input:
        page.goto("https://example.com");
        page.locator("button").click();
        
        Example output:
        1. Navigate to "https://example.com"
        2. Click the button element
        
        INPUT: {text}
        OUTPUT: Natural language instructions
        """
        
        system_message_prompt = SystemMessagePromptTemplate.from_template(template)
        human_template = "{text}"
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
        chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
        
        return LLMChain(llm=llm, prompt=chat_prompt)
    
    def _create_text_to_code_chain(self):
        """Create LLM chain for natural language to code conversion"""
        llm = Ollama(
            model=self.config["model_name"].get(),
            base_url=self.config["api_base"].get(),
            timeout=self.config["timeout"].get()
        )
        
        template = """
        You are a professional Playwright code generator. Convert natural language to valid Python code.
        
        Important guidelines:
        1. Output pure Python code with no explanations or comments
        2. Include necessary imports: from playwright.sync_api import sync_playwright
        3. Wrap code in a run(playwright) function
        4. Use proper indentation (4 spaces)
        5. Add error handling where appropriate
        6. Use expect assertions for validations
        7. Include context and browser cleanup
        
        Example input:
        "Go to https://example.com and click the 'Submit' button"
        
        Example output:
        def run(playwright):
            browser = playwright.chromium.launch(headless=False)
            context = browser.new_context()
            page = context.new_page()
            page.goto("https://example.com")
            page.get_by_text("Submit").click()
            context.close()
            browser.close()
        
        INPUT: {text}
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
        title_frame = ttk.Frame(self.root)
        title_frame.pack(fill=tk.X, pady=10)
        
        title_label = ttk.Label(
            title_frame, 
            text="Playwright Code Interpreter", 
            font=("Arial", 16, "bold")
        )
        title_label.pack(side=tk.LEFT, padx=20)
        
        # Create main frame
        main_frame = ttk.Notebook(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Workspace tab
        workspace_frame = ttk.Frame(main_frame)
        main_frame.add(workspace_frame, text="Workspace")
        
        # History tab
        history_frame = ttk.Frame(main_frame)
        main_frame.add(history_frame, text="History")
        
        # Create workspace layout
        input_frame = ttk.LabelFrame(workspace_frame, text="Input")
        input_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Separator between code input and natural language input
        input_splitter = ttk.Frame(input_frame, height=5)
        input_splitter.pack(fill=tk.X, pady=5)
        
        # Code input
        code_frame = ttk.LabelFrame(input_frame, text="Playwright Code Input")
        code_frame.pack(fill=tk.BOTH, expand=True, side=tk.LEFT, padx=(0, 5), pady=5)
        
        self.code_input = scrolledtext.ScrolledText(
            code_frame, 
            wrap=tk.WORD, 
            width=40, 
            height=15, 
            font=("Consolas", 10)
        )
        self.code_input.pack(fill=tk.BOTH, expand=True, pady=5, padx=5)
        
        # Natural language input
        nl_frame = ttk.LabelFrame(input_frame, text="Natural Language Description")
        nl_frame.pack(fill=tk.BOTH, expand=True, side=tk.RIGHT, padx=(5, 0), pady=5)
        
        self.natural_language = scrolledtext.ScrolledText(
            nl_frame, 
            wrap=tk.WORD, 
            width=40, 
            height=15, 
            font=("Arial", 10)
        )
        self.natural_language.pack(fill=tk.BOTH, expand=True, pady=5, padx=5)
        
        # Button frame
        button_frame = ttk.Frame(workspace_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        # Function buttons
        self.code_to_text_btn = ttk.Button(
            button_frame, 
            text="Code → Natural Language", 
            command=self._convert_code_to_text,
            style="Accent.TButton"
        )
        self.code_to_text_btn.pack(side=tk.LEFT, padx=5)
        
        self.text_to_code_btn = ttk.Button(
            button_frame, 
            text="Natural Language → Code", 
            command=self._convert_text_to_code,
            style="Accent.TButton"
        )
        self.text_to_code_btn.pack(side=tk.LEFT, padx=5)
        
        self.format_code_btn = ttk.Button(
            button_frame, 
            text="Format Code", 
            command=self._format_code,
            style="Accent.TButton"
        )
        self.format_code_btn.pack(side=tk.LEFT, padx=5)
        
        self.execute_btn = ttk.Button(
            button_frame, 
            text="Execute Code", 
            command=self._execute_code,
            style="Accent.TButton"
        )
        self.execute_btn.pack(side=tk.LEFT, padx=5)
        
        self.load_example_btn = ttk.Button(
            button_frame, 
            text="Load Example", 
            command=self._load_example,
            style="Accent.TButton"
        )
        self.load_example_btn.pack(side=tk.LEFT, padx=5)
        
        self.clear_all_btn = ttk.Button(
            button_frame, 
            text="Clear All", 
            command=self._clear_all,
            style="Accent.TButton"
        )
        self.clear_all_btn.pack(side=tk.LEFT, padx=5)
        
        # Code output frame
        output_frame = ttk.LabelFrame(workspace_frame, text="Playwright Code Output")
        output_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.code_output = scrolledtext.ScrolledText(
            output_frame, 
            wrap=tk.WORD, 
            width=80, 
            height=10, 
            font=("Consolas", 10)
        )
        self.code_output.pack(fill=tk.BOTH, expand=True, pady=5, padx=5)
        
        # Execution result frame
        result_frame = ttk.LabelFrame(workspace_frame, text="Execution Result")
        result_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.result_output = scrolledtext.ScrolledText(
            result_frame, 
            wrap=tk.WORD, 
            width=80, 
            height=5, 
            font=("Consolas", 10)
        )
        self.result_output.pack(fill=tk.BOTH, expand=True, pady=5, padx=5)
        
        # Status bar
        self.status_bar = ttk.Label(
            self.root, 
            text="Ready", 
            relief=tk.SUNKEN, 
            anchor=tk.W
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Configure styles
        self.style.configure("Accent.TButton", font=("Arial", 10, "bold"))
    
    def _setup_config_panel(self):
        """Set up configuration panel"""
        config_frame = ttk.LabelFrame(self.root, text="Configuration")
        config_frame.pack(fill=tk.X, padx=20, pady=5)
        
        # First row of configuration
        row1_frame = ttk.Frame(config_frame)
        row1_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Headless mode option
        ttk.Checkbutton(
            row1_frame,
            text="Run Playwright in headless mode",
            variable=self.config["headless"]
        ).pack(side=tk.LEFT, padx=(0, 20))
        
        # Timeout setting
        ttk.Label(row1_frame, text="LLM Timeout (seconds):").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Entry(row1_frame, textvariable=self.config["timeout"], width=5).pack(side=tk.LEFT)
        
        # Second row of configuration
        row2_frame = ttk.Frame(config_frame)
        row2_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Model configuration
        ttk.Label(row2_frame, text="Ollama Model:").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Entry(row2_frame, textvariable=self.config["model_name"], width=30).pack(side=tk.LEFT, padx=(0, 20))
        
        # API Base configuration
        ttk.Label(row2_frame, text="Ollama API Base:").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Entry(row2_frame, textvariable=self.config["api_base"], width=40).pack(side=tk.LEFT)
    
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
        except TimeoutError:
            messagebox.showerror("Timeout", "LLM took too long to respond. Please try again.")
            self.status_bar.config(text="Ready")
        except ConnectionError:
            messagebox.showerror("Connection Error", f"Could not connect to Ollama API at {API_BASE}")
            self.status_bar.config(text="Ready")
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
            cleaned_code = self._cleanse_code_output(result)
            
            # Apply headless mode configuration
            if self.config["headless"].get():
                cleaned_code = cleaned_code.replace(
                    "launch(headless=False)", 
                    "launch(headless=True)"
                )
            
            self.code_output.delete("1.0", tk.END)
            self.code_output.insert(tk.END, cleaned_code)
            self.status_bar.config(text="Code generation completed")
        except TimeoutError:
            messagebox.showerror("Timeout", "LLM took too long to respond. Please try again.")
            self.status_bar.config(text="Ready")
        except ConnectionError:
            messagebox.showerror("Connection Error", f"Could not connect to Ollama API at {self.config['api_base'].get()}")
            self.status_bar.config(text="Ready")
        except Exception as e:
            messagebox.showerror("Error", f"Error generating code: {str(e)}")
            self.status_bar.config(text="Ready")
    
    def _cleanse_code_output(self, code):
        """Cleanse LLM output to ensure valid Python code"""
        # Remove common LLM response prefixes and markers
        prefixes = ["```python", "```", "python", "# Code:", "# Output:"]
        for prefix in prefixes:
            if code.startswith(prefix):
                code = code[len(prefix):].lstrip()
        
        if code.endswith("```"):
            code = code[:-3].rstrip()
        
        # Ensure code contains necessary structure
        if "def run(playwright):" not in code:
            # If no run function, try to wrap code
            code = f"def run(playwright):\n    # Generated code\n{self._indent_code(code)}\n    context.close()\n    browser.close()"
        
        # Ensure necessary imports are included
        if "from playwright.sync_api import" not in code:
            code = "from playwright.sync_api import sync_playwright\n\n" + code
        
        # Ensure there's a with statement to call the run function
        if "with sync_playwright() as playwright:" not in code:
            code += "\n\nwith sync_playwright() as playwright:\n    run(playwright)"
        
        return code
    
    def _indent_code(self, code, indent=4):
        """Indent code block"""
        lines = code.split("\n")
        indented_lines = [" " * indent + line for line in lines]
        return "\n".join(indented_lines)
    
    def _format_code(self):
        """Format code output"""
        code = self.code_output.get("1.0", tk.END).strip()
        if not code:
            return
        
        try:
            # Try to format code using black
            result = subprocess.run(
                ["black", "-q", "-c", code],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                formatted_code = result.stdout
                self.code_output.delete("1.0", tk.END)
                self.code_output.insert(tk.END, formatted_code)
                messagebox.showinfo("Success", "Code formatted successfully!")
            else:
                messagebox.showerror("Formatting Error", f"Failed to format code:\n{result.stderr}")
        except FileNotFoundError:
            messagebox.showerror("Dependency Missing", "Please install black:\n\npip install black")
        except Exception as e:
            messagebox.showerror("Error", f"Error formatting code:\n{str(e)}")
    
    def _execute_code(self):
        """Execute Playwright code"""
        code = self.code_output.get("1.0", tk.END).strip()
        if not code:
            messagebox.showwarning("Warning", "Please generate Playwright code first")
            return
        
        # Check if code contains necessary Playwright imports
        if "playwright.sync_api" not in code:
            messagebox.showwarning("Invalid Code", "Generated code does not contain Playwright imports")
            return
        
        temp_file_path = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
                f.write(code)
                temp_file_path = f.name
            
            self.status_bar.config(text="Executing code...")
            self.root.update()
            
            # Clear output
            self.result_output.delete("1.0", tk.END)
            
            # Detect Python executable
            python_executable = self._detect_python_executable()
            if not python_executable:
                messagebox.showerror("Python Not Found", "Could not find a valid Python installation")
                return
            
            # Check if Playwright is installed
            if not self._check_playwright_installed(python_executable):
                if messagebox.askyesno("Install Playwright", "Playwright is not installed. Install it now?"):
                    self._install_playwright(python_executable)
                else:
                    self.status_bar.config(text="Ready")
                    return
            
            # Execute code
            self._run_code_with_realtime_output(python_executable, temp_file_path)
            
        except Exception as e:
            messagebox.showerror("Execution Error", f"Failed to execute code: {str(e)}")
            self.status_bar.config(text="Ready")
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)
    
    def _detect_python_executable(self):
        """Detect available Python executable"""
        candidates = [
            sys.executable,
            "python3",
            "python",
            "py"  # Windows Python Launcher
        ]
        
        for candidate in candidates:
            try:
                result = subprocess.run(
                    [candidate, "--version"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0 and "Python" in result.stdout:
                    return candidate
            except:
                continue
        
        return None
    
    def _check_playwright_installed(self, python_executable):
        """Check if Playwright is installed"""
        try:
            result = subprocess.run(
                [python_executable, "-c", "import playwright; print(playwright.__version__)"],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.returncode == 0
        except:
            return False
    
    def _install_playwright(self, python_executable):
        """Install Playwright and its browsers"""
        try:
            self.result_output.delete("1.0", tk.END)
            self.result_output.insert(tk.END, "Installing Playwright...\n")
            
            # Install Playwright library
            install_process = subprocess.Popen(
                [python_executable, "-m", "pip", "install", "playwright"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            for line in iter(install_process.stdout.readline, ''):
                self.result_output.insert(tk.END, line)
                self.result_output.see(tk.END)
                self.root.update_idletasks()
            
            install_process.wait()
            
            if install_process.returncode != 0:
                self.result_output.insert(tk.END, "\nFailed to install Playwright library.\n")
                return False
            
            # Install browsers
            self.result_output.insert(tk.END, "\nInstalling browsers...\n")
            
            browser_process = subprocess.Popen(
                [python_executable, "-m", "playwright", "install"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            for line in iter(browser_process.stdout.readline, ''):
                self.result_output.insert(tk.END, line)
                self.result_output.see(tk.END)
                self.root.update_idletasks()
            
            browser_process.wait()
            
            if browser_process.returncode != 0:
                self.result_output.insert(tk.END, "\nFailed to install browsers.\n")
                return False
            
            self.result_output.insert(tk.END, "\nPlaywright installed successfully!\n")
            return True
        except Exception as e:
            self.result_output.insert(tk.END, f"\nError installing Playwright: {str(e)}\n")
            return False
    
    def _run_code_with_realtime_output(self, python_executable, file_path):
        """Execute code and display real-time output"""
        process = subprocess.Popen(
            [python_executable, file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # Display output in real-time
        self.result_output.delete("1.0", tk.END)
        for line in iter(process.stdout.readline, ''):
            self.result_output.insert(tk.END, line)
            self.result_output.see(tk.END)  # Auto-scroll
            self.root.update_idletasks()  # Update UI
        
        process.wait()
        
        if process.returncode == 0:
            self.result_output.insert(tk.END, "\n\nExecution completed successfully!")
        else:
            self.result_output.insert(tk.END, f"\n\nExecution failed with return code {process.returncode}")
        
        self.status_bar.config(text="Execution completed")
    
    def _run_command(self, file_path):
        """直接用python命令执行生成的py文件"""
        import subprocess
        try:
            result = subprocess.run([
                "python", file_path
            ], capture_output=True, text=True, timeout=120)
            if result.returncode == 0:
                self.result_output.insert(tk.END, "执行成功！\n\n标准输出:\n" + result.stdout)
            else:
                self.result_output.insert(tk.END, f"执行失败 (返回码: {result.returncode})\n\n标准输出:\n{result.stdout}\n\n错误输出:\n{result.stderr}")
        except subprocess.TimeoutExpired:
            self.result_output.insert(tk.END, "执行超时 (超过120秒)")
        except FileNotFoundError:
            self.result_output.insert(tk.END, "未找到python命令，请确保已正确安装Python环境")
        except Exception as e:
            self.result_output.insert(tk.END, f"执行异常: {str(e)}")
    
    def _load_example(self):
        """Load example data"""
        # Clear all fields
        self._clear_all()
        
        # Load Python Playwright example
        example_code = """from playwright.sync_api import sync_playwright, expect

def run(playwright):
    browser = playwright.chromium.launch(headless=False)
    context = browser.new_context()
    page = context.new_page()
    
    # Open website
    page.goto('https://www.saucedemo.com/')
    
    # Login
    page.locator('[data-test="username"]').fill('standard_user')
    page.locator('[data-test="password"]').fill('secret_sauce')
    page.locator('[data-test="login-button"]').click()
    
    # Verify login success
    expect(page.locator('[data-test="title"]')).to_have_text('Products')
    
    # Add item to cart
    page.locator('[data-test="add-to-cart-sauce-labs-backpack"]').click()
    
    # View cart
    page.locator('[data-test="shopping-cart-link"]').click()
    
    # Checkout
    page.locator('[data-test="checkout"]').click()
    
    # Fill in information
    page.locator('[data-test="firstName"]').fill('John')
    page.locator('[data-test="lastName"]').fill('Doe')
    page.locator('[data-test="postalCode"]').fill('12345')
    page.locator('[data-test="continue"]').click()
    
    # Complete order
    page.locator('[data-test="finish"]').click()
    
    # Verify order completion
    expect(page.locator('.complete-header')).to_have_text('Thank you for your order!')
    
    # Logout
    page.get_by_role('button', name='Open Menu').click()
    page.locator('[data-test="logout-sidebar-link"]').click()
    
    # Verify logout
    expect(page.locator('[data-test="login-button"]')).to_be_visible()
    
    # Close browser
    context.close()
    browser.close()

with sync_playwright() as playwright:
    run(playwright)
"""
        
        self.code_input.insert(tk.END, example_code)
        self.status_bar.config(text="Example data loaded (Python Playwright)")
    
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
