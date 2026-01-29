# ui_main.py
import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox
from config import Config
from database.vector_store import VectorStore
from database.session_manager import SessionManager
from agent.personal_agent import PersonalAgent
from helper.speechtotext import voice_search

class PersonalAIGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Personal AI Knowledge Agent")
        self.root.geometry("700x500")

        # Initialize agent
        Config.create_dirs()
        self.vector_store = VectorStore(str(Config.VECTOR_DB_PATH))
        self.session_manager = SessionManager(str(Config.METADATA_DB_PATH))
        self.agent = PersonalAgent(self.vector_store, self.session_manager)

        # Mode selection
        self.mode_var = tk.StringVar(value="add")
        tk.Label(root, text="Mode:").pack()
        tk.OptionMenu(root, self.mode_var, "add", "query", "chat", "stats").pack()

        # Input type
        self.input_var = tk.StringVar(value="text")
        tk.Label(root, text="Input Type:").pack()
        tk.OptionMenu(root, self.input_var, "text", "voice").pack()

        # Text input
        tk.Label(root, text="Text / Question:").pack()
        self.text_input = scrolledtext.ScrolledText(root, height=5)
        self.text_input.pack(fill=tk.X, padx=5, pady=5)

        # File selection
        self.file_path = tk.StringVar()
        tk.Button(root, text="Select File", command=self.select_file).pack()
        tk.Label(root, textvariable=self.file_path).pack()

        # Output area
        tk.Label(root, text="Output:").pack()
        self.output_area = scrolledtext.ScrolledText(root, height=15)
        self.output_area.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Action button
        tk.Button(root, text="Run", command=self.run_mode, bg="lightblue").pack(pady=10)

    def select_file(self):
        file = filedialog.askopenfilename()
        if file:
            self.file_path.set(file)

    def run_mode(self):
        mode = self.mode_var.get()
        input_type = self.input_var.get()
        text = self.text_input.get("1.0", tk.END).strip()
        file = self.file_path.get() or None

        self.output_area.delete("1.0", tk.END)

        try:
            if mode == "stats":
                stats = self.agent.get_stats()
                for k, v in stats.items():
                    self.output_area.insert(tk.END, f"{k}: {v}\n")

            elif mode == "add":
                if input_type == "voice":
                    text = voice_search()
                    if not text:
                        messagebox.showerror("Error", "No speech recognized.")
                        return
                elif file:
                    with open(file, "r", encoding="utf-8") as f:
                        text = f.read()

                if not text:
                    messagebox.showerror("Error", "No input provided.")
                    return

                doc_ids = self.agent.add_to_knowledge_base(
                    text, source="manual", metadata={"input_type": input_type, "file": file}
                )
                self.output_area.insert(tk.END, f"Added {len(doc_ids)} chunks to knowledge base.\n")

            elif mode == "query":
                if input_type == "voice":
                    text = voice_search()
                    if not text:
                        messagebox.showerror("Error", "No speech recognized.")
                        return

                if not text:
                    messagebox.showerror("Error", "No question provided.")
                    return

                result = self.agent.query(text)
                self.output_area.insert(tk.END, f"Question: {result['question']}\n\n")
                for i, (chunk, distance) in enumerate(zip(result['context'], result['distances']), 1):
                    self.output_area.insert(tk.END, f"[{i}] (similarity: {1 - distance:.3f})\n{chunk[:300]}...\n\n")

            elif mode == "chat":
                session_id = self.agent.start_session({"mode": "chat", "input_type": input_type})
                user_input = text
                if input_type == "voice":
                    user_input = voice_search()
                    if not user_input:
                        messagebox.showerror("Error", "No speech recognized.")
                        return
                response = self.agent.chat(user_input)
                self.output_area.insert(tk.END, f"You: {user_input}\n")
                self.output_area.insert(tk.END, f"Assistant: {response}\n")

        except Exception as e:
            messagebox.showerror("Error", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    app = PersonalAIGUI(root)
    root.mainloop()
