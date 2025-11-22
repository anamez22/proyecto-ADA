"""
Interfaz gráfica para análisis de complejidad asintótica de algoritmos C
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import json
from neural_network import ComplexityAnalyzerNN, CodeTokenizer
from graphics_analyzer import CodeAnalyzer, TrainingGraphs
import os
import sys
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np

class ComplexityAnalyzerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Analizador de Complejidad Asintótica - C Language")
        self.root.geometry("1200x800")
        self.root.configure(bg="#000000")
        
        # Variables
        self.model_loaded = False
        self.analyzer = None
        self.last_prediction = None
        
        # Estilos
        self.setup_styles()
        
        # UI
        self.create_widgets()
        
        # Cargar modelo automáticamente
        self.load_model_automatic()
    
    def setup_styles(self):
        """Configura estilos de la interfaz"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Colores personalizados
        bg_color = "#4d4444"
        fg_color = '#e0e0e0'
        accent_color = "#055A45"
        
        style.configure('TFrame', background=bg_color)
        style.configure('TLabel', background=bg_color, foreground=fg_color)
        style.configure('TButton', background=accent_color, foreground=fg_color)
        style.configure('TText', background='#1e1e1e', foreground=fg_color)
        
        # Estilos personalizados
        style.configure('Header.TLabel', font=('Arial', 14, 'bold'), foreground="#01b4d8")
        style.configure('Title.TLabel', font=('Arial', 18, 'bold'), foreground='#00d4ff')
        style.configure('Result.TLabel', font=('Arial', 12, 'bold'), foreground='#00ff00')
        style.configure('Error.TLabel', font=('Arial', 10), foreground="#f55353")
    
    def create_widgets(self):
        """Crea los widgets de la interfaz con tabs"""
        # Frame principal
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # ==== Título ====
        title_label = ttk.Label(main_frame, text="🧠 Analizador de Complejidad Asintótica", 
                                style='Title.TLabel')
        title_label.pack(pady=(0, 20))
        
        # ==== Notebook (Tabs) ====
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Tab 1: Análisis Principal
        self.create_analysis_tab()
        
        # Tab 2: Estadísticas del Código
        self.create_statistics_tab()
        
        # Tab 3: Gráficas de Entrenamiento
        self.create_training_tab()
    
    def create_analysis_tab(self):
        """Crea la pestaña de análisis principal"""
        analysis_frame = ttk.Frame(self.notebook)
        self.notebook.add(analysis_frame, text="📊 Análisis Principal")
        
        # ==== Frame Superior: Entrada de Código ====
        input_frame = ttk.LabelFrame(analysis_frame, text="Código C", padding=10)
        input_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Botones de entrada
        button_frame = ttk.Frame(input_frame)
        button_frame.pack(fill=tk.X, pady=(0, 10))
        
        load_btn = ttk.Button(button_frame, text="📂 Cargar Archivo", 
                              command=self.load_file)
        load_btn.pack(side=tk.LEFT, padx=5)
        
        clear_btn = ttk.Button(button_frame, text="🗑️ Limpiar", 
                               command=self.clear_code)
        clear_btn.pack(side=tk.LEFT, padx=5)
        
        analyze_btn = ttk.Button(button_frame, text="🔍 Analizar", 
                                 command=self.analyze_code)
        analyze_btn.pack(side=tk.LEFT, padx=5)
        
        status_label = ttk.Label(button_frame, text="Estado: ", 
                                 foreground='#ffff00')
        status_label.pack(side=tk.RIGHT, padx=5)
        self.status_label = status_label
        
        # Área de texto para código
        self.code_text = scrolledtext.ScrolledText(
            input_frame,
            height=12,
            width=80,
            font=('Courier New', 10),
            bg='#1e1e1e',
            fg='#d4d4d4',
            insertbackground='white',
            wrap=tk.WORD
        )
        self.code_text.pack(fill=tk.BOTH, expand=True)
        
        # ==== Frame Inferior: Resultados ====
        result_frame = ttk.LabelFrame(analysis_frame, text="Resultados del Análisis", padding=10)
        result_frame.pack(fill=tk.BOTH, expand=True)
        
        # Frame para complejidad detectada
        complexity_frame = ttk.Frame(result_frame)
        complexity_frame.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Label(complexity_frame, text="Complejidad Detectada:", 
                  style='Header.TLabel').pack(side=tk.LEFT, padx=(0, 10))
        
        self.complexity_result = ttk.Label(complexity_frame, text="---", 
                                           style='Result.TLabel', font=('Arial', 16, 'bold'),
                                           foreground='#ffff00')
        self.complexity_result.pack(side=tk.LEFT)
        
        self.confidence_result = ttk.Label(complexity_frame, text="", 
                                           foreground='#00ff00')
        self.confidence_result.pack(side=tk.LEFT, padx=(10, 0))
        
        # ==== Gráfico de probabilidades ====
        chart_frame = ttk.LabelFrame(result_frame, text="Probabilidades por Complejidad", padding=10)
        chart_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Canvas para gráfico
        self.chart_canvas = tk.Canvas(
            chart_frame,
            height=120,
            bg='#1e1e1e',
            highlightthickness=0
        )
        self.chart_canvas.pack(fill=tk.BOTH, expand=True)
        
        # ==== Panel de Tokens ====
        tokens_frame = ttk.LabelFrame(result_frame, text="Tokens Detectados", padding=10)
        tokens_frame.pack(fill=tk.X)
        
        self.tokens_text = scrolledtext.ScrolledText(
            tokens_frame,
            height=3,
            width=80,
            font=('Courier New', 9),
            bg='#1e1e1e',
            fg='#d4d4d4',
            wrap=tk.WORD
        )
        self.tokens_text.pack(fill=tk.BOTH, expand=True)
    
    def create_statistics_tab(self):
        """Crea la pestaña de estadísticas del código"""
        stats_frame = ttk.Frame(self.notebook)
        self.notebook.add(stats_frame, text="📈 Estadísticas del Código")
        
        # Frame para botón de análisis
        button_frame = ttk.Frame(stats_frame)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        stats_btn = ttk.Button(button_frame, text="📊 Calcular Estadísticas", 
                               command=self.calculate_statistics)
        stats_btn.pack(side=tk.LEFT, padx=5)
        
        # Frame para tabla de estadísticas
        table_frame = ttk.LabelFrame(stats_frame, text="Métricas del Código", padding=10)
        table_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Treeview para tabla
        columns = ('Métrica', 'Valor')
        self.stats_tree = ttk.Treeview(table_frame, columns=columns, height=13, show='headings')
        
        self.stats_tree.column('Métrica', width=200, anchor='w')
        self.stats_tree.column('Valor', width=100, anchor='center')
        
        self.stats_tree.heading('Métrica', text='Métrica')
        self.stats_tree.heading('Valor', text='Valor')
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.stats_tree.yview)
        self.stats_tree.configure(yscroll=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.stats_tree.pack(fill=tk.BOTH, expand=True)
    
    def create_training_tab(self):
        """Crea la pestaña de gráficas de entrenamiento"""
        training_frame = ttk.Frame(self.notebook)
        self.notebook.add(training_frame, text="📚 Gráficas de Entrenamiento")
        
        # Frame para selector de gráfica
        selector_frame = ttk.Frame(training_frame)
        selector_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(selector_frame, text="Seleccionar gráfica:", font=('Arial', 10)).pack(side=tk.LEFT, padx=5)
        
        self.graph_var = tk.StringVar(value="loss")
        graphs = [
            ("loss", "Pérdida de Entrenamiento"),
            ("accuracy", "Precisión"),
            ("confusion", "Matriz de Confusión")
        ]
        
        for value, label in graphs:
            ttk.Radiobutton(selector_frame, text=label, variable=self.graph_var, 
                           value=value, command=self.update_training_graph).pack(side=tk.LEFT, padx=5)
        
        # Frame para gráfica
        graph_frame = ttk.LabelFrame(training_frame, text="Visualización", padding=10)
        graph_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.graph_canvas = tk.Canvas(graph_frame, bg='#1e1e1e', highlightthickness=0)
        self.graph_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Información del modelo
        info_frame = ttk.LabelFrame(training_frame, text="Información del Modelo", padding=10)
        info_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        info_text = (
            "Arquitectura: MLP (256 → 128 → 64 → 32 neuronas)\n"
            "Características: 21 discriminantes de código\n"
            "Precisión en entrenamiento: 96.15%\n"
            "Precisión en test: 100% (7/7 casos)\n"
            "Precisión global: 97% (32/33 casos)\n"
            "Optimizador: Adam | Early stopping: 50 épocas\n"
            "Iteraciones a convergencia: 72 épocas"
        )
        
        ttk.Label(info_frame, text=info_text, font=('Courier', 9), justify=tk.LEFT).pack(fill=tk.BOTH)
        
        # Dibujar gráfica inicial
        self.update_training_graph()
    
    def calculate_statistics(self):
        """Calcula y muestra estadísticas del código"""
        code = self.code_text.get(1.0, tk.END).strip()
        
        if not code:
            messagebox.showwarning("Advertencia", "Por favor, ingrese código C")
            return
        
        try:
            # Analizar código
            analyzer = CodeAnalyzer(code)
            stats = analyzer.analyze()
            
            # Limpiar tabla
            for item in self.stats_tree.get_children():
                self.stats_tree.delete(item)
            
            # Llenar tabla
            for metric, value in analyzer.get_formatted_stats():
                self.stats_tree.insert('', 'end', values=(metric, value))
            
            messagebox.showinfo("Éxito", "Estadísticas calculadas correctamente")
        except Exception as e:
            messagebox.showerror("Error", f"Error al calcular estadísticas: {str(e)}")
    
    def update_training_graph(self):
        """Actualiza la gráfica de entrenamiento según la selección"""
        graph_type = self.graph_var.get()
        
        # Limpiar canvas
        for widget in self.graph_canvas.winfo_children():
            widget.destroy()
        
        # Obtener datos
        graphs = TrainingGraphs()
        
        if graph_type == "loss":
            self.draw_loss_graph(graphs)
        elif graph_type == "accuracy":
            self.draw_accuracy_graph(graphs)
        elif graph_type == "confusion":
            self.draw_confusion_matrix(graphs)
    
    def draw_loss_graph(self, graphs_obj):
        """Dibuja gráfica de pérdida"""
        data = graphs_obj.generate_training_curves()
        
        fig = Figure(figsize=(10, 4), dpi=80, facecolor='#4d4444')
        ax = fig.add_subplot(111)
        ax.set_facecolor('#1e1e1e')
        
        ax.plot(data['epochs'], data['train_loss'], label='Pérdida Entrenamiento', 
                color='#00d4ff', linewidth=2, marker='o', markersize=3)
        ax.plot(data['epochs'], data['test_loss'], label='Pérdida Test', 
                color='#ff6b6b', linewidth=2, marker='s', markersize=3)
        
        ax.set_xlabel('Época', color='white', fontsize=10)
        ax.set_ylabel('Pérdida (MSE)', color='white', fontsize=10)
        ax.set_title('Curva de Pérdida del Modelo', color='white', fontsize=12, fontweight='bold')
        ax.legend(facecolor='#4d4444', edgecolor='white', labelcolor='white')
        ax.grid(True, alpha=0.2, color='white')
        ax.tick_params(colors='white')
        
        canvas = FigureCanvasTkAgg(fig, master=self.graph_canvas)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def draw_accuracy_graph(self, graphs_obj):
        """Dibuja gráfica de precisión"""
        data = graphs_obj.generate_training_curves()
        
        fig = Figure(figsize=(10, 4), dpi=80, facecolor='#4d4444')
        ax = fig.add_subplot(111)
        ax.set_facecolor('#1e1e1e')
        
        ax.plot(data['epochs'], data['accuracy'], label='Precisión', 
                color='#00ff00', linewidth=2.5, marker='D', markersize=4)
        
        ax.axhline(y=0.965, color='#ffff00', linestyle='--', linewidth=2, label='Accuracy Objetivo (96.5%)')
        ax.axhline(y=0.97, color='#ff6b6b', linestyle='--', linewidth=2, label='Accuracy Global (97%)')
        
        ax.set_xlabel('Época', color='white', fontsize=10)
        ax.set_ylabel('Precisión', color='white', fontsize=10)
        ax.set_title('Curva de Precisión del Modelo', color='white', fontsize=12, fontweight='bold')
        ax.set_ylim([0, 1])
        ax.legend(facecolor='#4d4444', edgecolor='white', labelcolor='white')
        ax.grid(True, alpha=0.2, color='white')
        ax.tick_params(colors='white')
        
        canvas = FigureCanvasTkAgg(fig, master=self.graph_canvas)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def draw_confusion_matrix(self, graphs_obj):
        """Dibuja matriz de confusión"""
        data = graphs_obj.generate_confusion_data()
        
        fig = Figure(figsize=(8, 8), dpi=80, facecolor='#4d4444')
        ax = fig.add_subplot(111)
        ax.set_facecolor('#1e1e1e')
        
        matrix = np.array(data['matrix'])
        im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')
        
        # Etiquetas
        complexities = data['complexities']
        ax.set_xticks(range(len(complexities)))
        ax.set_yticks(range(len(complexities)))
        ax.set_xticklabels(complexities, rotation=45, ha='right', color='white')
        ax.set_yticklabels(complexities, color='white')
        
        # Valores en las celdas
        for i in range(len(complexities)):
            for j in range(len(complexities)):
                text = ax.text(j, i, matrix[i, j], ha="center", va="center",
                             color="white", fontsize=10, fontweight='bold')
        
        ax.set_title(f"Matriz de Confusión\nAccuracy: {data['accuracy']:.1%}", 
                    color='white', fontsize=12, fontweight='bold', pad=20)
        ax.set_xlabel('Predicción', color='white', fontsize=10)
        ax.set_ylabel('Real', color='white', fontsize=10)
        
        fig.colorbar(im, ax=ax, label='Cantidad')
        
        canvas = FigureCanvasTkAgg(fig, master=self.graph_canvas)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def load_model_automatic(self):
        """Carga el modelo automáticamente en un thread"""
        def load():
            try:
                self.update_status("Cargando modelo...")
                self.root.update()
                
                # Registrar CodeTokenizer en __main__
                sys.modules['__main__'].CodeTokenizer = CodeTokenizer
                
                model_path = "complexity_model.pkl"
                tokenizer_path = "tokenizer.pkl"
                scaler_path = "scaler.pkl"
                
                if os.path.exists(model_path) and os.path.exists(tokenizer_path) and os.path.exists(scaler_path):
                    self.analyzer = ComplexityAnalyzerNN()
                    self.analyzer.load_model(model_path, tokenizer_path, scaler_path)
                    self.model_loaded = True
                    self.update_status("Modelo cargado [OK]")
                    messagebox.showinfo("Exito", "Modelo entrenado cargado correctamente")
                else:
                    self.update_status("Modelo no encontrado")
                    messagebox.showwarning("Modelo no encontrado", 
                                          f"Por favor, entrene el modelo primero:\n"
                                          f"1. Ejecute: python dataset_generator.py\n"
                                          f"2. Ejecute: python neural_network.py")
            except Exception as e:
                self.update_status("Error al cargar modelo")
                messagebox.showerror("Error", f"Error al cargar modelo: {str(e)}")
        
        thread = threading.Thread(target=load, daemon=True)
        thread.start()
    
    def load_file(self):
        """Carga código desde un archivo"""
        file_path = filedialog.askopenfilename(
            title="Seleccionar archivo C",
            filetypes=[("C files", "*.c"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    code = f.read()
                self.code_text.delete(1.0, tk.END)
                self.code_text.insert(1.0, code)
                self.update_status("Archivo cargado ✓")
            except Exception as e:
                messagebox.showerror("Error", f"Error al cargar archivo: {str(e)}")
    
    def clear_code(self):
        """Limpia el área de texto"""
        self.code_text.delete(1.0, tk.END)
        self.complexity_result.config(text="---")
        self.confidence_result.config(text="")
        self.tokens_text.delete(1.0, tk.END)
        self.chart_canvas.delete("all")
        self.last_prediction = None
        self.update_status("Listo")
    
    def analyze_code(self):
        """Analiza el código C"""
        if not self.model_loaded:
            messagebox.showerror("Error", "El modelo no está cargado")
            return
        
        code = self.code_text.get(1.0, tk.END).strip()
        
        if not code:
            messagebox.showwarning("Advertencia", "Por favor, ingrese código C")
            return
        
        def analyze():
            try:
                self.update_status("Analizando...")
                self.root.update()
                
                # Analizar
                result = self.analyzer.predict_complexity(code)
                self.last_prediction = result
                
                # Mostrar resultados
                self.show_results(result)
                self.update_status("Análisis completado [OK]")
                
            except ValueError as e:
                # Error de validación (lenguaje incorrecto)
                self.update_status("Error: Lenguaje no soportado")
                messagebox.showerror("Error de Validación", str(e))
            except Exception as e:
                self.update_status("Error en análisis")
                messagebox.showerror("Error", f"Error durante análisis: {str(e)}")
        
        thread = threading.Thread(target=analyze, daemon=True)
        thread.start()
    
    def show_results(self, result):
        """Muestra los resultados del análisis"""
        # Complejidad y confianza
        complexity = result['complexity']
        confidence = result['confidence']
        
        self.complexity_result.config(text=complexity)
        self.confidence_result.config(text=f"Confianza: {confidence:.1%}")
        
        # Tokens
        tokens = result['tokens']
        tokens_str = ", ".join(tokens[:30])  # Mostrar primeros 30
        if len(tokens) > 30:
            tokens_str += f", ... (+{len(tokens) - 30} más)"
        self.tokens_text.delete(1.0, tk.END)
        self.tokens_text.insert(1.0, tokens_str)
        
        # Gráfico de probabilidades
        self.draw_probability_chart(result['probabilities'])
    
    def draw_probability_chart(self, probabilities):
        """Dibuja un gráfico de barras con las probabilidades"""
        self.chart_canvas.delete("all")
        
        complexities = list(probabilities.keys())
        probs = list(probabilities.values())
        
        canvas_width = self.chart_canvas.winfo_width()
        canvas_height = self.chart_canvas.winfo_height()
        
        if canvas_width < 2:
            canvas_width = 1000
        if canvas_height < 2:
            canvas_height = 150
        
        # Márgenes
        margin_left = 60
        margin_right = 20
        margin_top = 20
        margin_bottom = 40
        
        chart_width = canvas_width - margin_left - margin_right
        chart_height = canvas_height - margin_top - margin_bottom
        
        max_prob = max(probs) if probs else 1
        
        # Colores según confianza
        def get_color(prob):
            if prob > 0.5:
                return '#00ff00'  # Verde fuerte
            elif prob > 0.2:
                return '#ffff00'  # Amarillo
            else:
                return '#ff6b6b'  # Rojo
        
        # Dibujar barras
        bar_width = chart_width / len(complexities)
        
        for i, (complexity, prob) in enumerate(zip(complexities, probs)):
            x1 = margin_left + i * bar_width + 5
            x2 = x1 + bar_width - 10
            
            # Alto de la barra proporcional a probabilidad
            bar_height = (prob / max_prob) * chart_height if max_prob > 0 else 0
            
            y1 = margin_top + chart_height - bar_height
            y2 = margin_top + chart_height
            
            # Color
            color = get_color(prob)
            
            # Dibujar barra
            self.chart_canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline='white')
            
            # Etiqueta con porcentaje
            prob_text = f"{prob:.0%}"
            self.chart_canvas.create_text(
                (x1 + x2) / 2, y1 - 5,
                text=prob_text,
                font=('Arial', 8),
                fill='white'
            )
            
            # Etiqueta de complejidad
            self.chart_canvas.create_text(
                (x1 + x2) / 2, y2 + 15,
                text=complexity,
                font=('Arial', 8),
                fill='white'
            )
        
        # Eje Y
        self.chart_canvas.create_line(margin_left, margin_top, margin_left, 
                                     margin_top + chart_height, fill='white')
        self.chart_canvas.create_line(margin_left - 5, margin_top + chart_height, 
                                     canvas_width - margin_right, margin_top + chart_height, 
                                     fill='white')
    
    def update_status(self, message):
        """Actualiza el label de estado"""
        self.status_label.config(text=f"Estado: {message}")


def main():
    root = tk.Tk()
    gui = ComplexityAnalyzerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
