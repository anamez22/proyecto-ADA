"""
Red Neuronal para análisis de complejidad asintótica de algoritmos en C
Procesa código C tokenizado - Versión con scikit-learn
"""

import json
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import pickle
import os
import joblib


class CodeTokenizer:
    """Tokenizador especializado para código C"""
    
    def __init__(self):
        self.keywords = {
            'for', 'while', 'if', 'else', 'switch', 'case', 'break', 'continue',
            'return', 'int', 'char', 'float', 'double', 'void', 'struct',
            'array', 'malloc', 'free', 'recursion'
        }
        
        self.token_to_id = {}
        self.id_to_token = {}
        self.vocab_size = 0
        self.setup_vocabulary()
    
    def setup_vocabulary(self):
        """Configura el vocabulario básico"""
        tokens = ['<PAD>', '<UNK>', '<START>', '<END>']
        tokens.extend(sorted(self.keywords))
        operators = ['+', '-', '*', '/', '%', '++', '--', '==', '!=', '<', '>', '<=', '>=',
                     '&&', '||', '!', '&', '|', '^', '~', '=', '+=', '-=', '*=', '/=']
        tokens.extend(operators)
        tokens.extend(['NUMBER', 'IDENTIFIER', 'BRACKET', 'PAREN', 'BRACE', 'COMMA', 'SEMICOLON'])
        
        for idx, token in enumerate(tokens):
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token
        
        self.vocab_size = len(tokens)
    
    def tokenize_code(self, code):
        """Tokeniza código C"""
        code = re.sub(r'//.*?$', '', code, flags=re.MULTILINE)
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
        
        tokens = []
        patterns = [
            (r'\b(?:' + '|'.join(self.keywords) + r')\b', lambda m: m.group(0)),
            (r'\d+', lambda m: 'NUMBER'),
            (r'\+\+|--|==|!=|<=|>=|&&|\|\||[+\-*/%=&|^~!<>]', lambda m: m.group(0)),
            (r'[(){}\[\],;]', lambda m: {'(': 'PAREN', ')': 'PAREN', '{': 'BRACE', '}': 'BRACE', 
                                          '[': 'BRACKET', ']': 'BRACKET', ',': 'COMMA', ';': 'SEMICOLON'}.get(m.group(0), m.group(0))),
            (r'[a-zA-Z_]\w*', lambda m: 'IDENTIFIER'),
        ]
        
        idx = 0
        while idx < len(code):
            match = None
            for pattern, transform in patterns:
                regex = re.match(pattern, code[idx:])
                if regex:
                    token_str = regex.group(0)
                    token = transform(regex)
                    tokens.append(token)
                    idx += len(token_str)
                    match = True
                    break
            
            if not match:
                idx += 1
        
        return tokens
    
    def extract_features(self, code):
        """Extrae características del código para análisis - V6 Final"""
        tokens = self.tokenize_code(code)
        
        # Características base (13 originales que funcionaban)
        for_loops = tokens.count('for')
        while_loops = tokens.count('while')
        if_statements = tokens.count('if')
        recursive_calls = len(re.findall(r'\b\w+\s*\(\s*[^)]*\b\w+\b', code))
        nested_depth = self.calculate_nesting_depth(code)
        
        # Características que mejoraron el modelo (4 más)
        recursion_depth = self._measure_recursion_depth(code)
        nested_for_count = self._count_nested_for_loops(code)
        is_exponential_recursion = self._has_multiple_recursive_calls_in_function(code)
        branch_count = tokens.count('if') + tokens.count('switch') + tokens.count('case')
        
        # 3 características discriminantes
        halving_pattern = len(re.findall(r'/\s*2|>>\s*1|log', code, re.IGNORECASE))
        exponential_indicators = len(re.findall(r'\*\s*2|<<\s*1|pow.*2', code, re.IGNORECASE))
        triple_nested = self._count_triple_nested_loops(code)
        
        # NUEVA: Ratio de recursive calls - muy importante para O(2^n)
        # Si hay muchas llamadas recursivas, es probablemente O(2^n)
        recursive_call_ratio = recursive_calls / max(for_loops + while_loops + 1, 1)
        
        features = {
            'token_count': len(tokens),
            'for_loops': for_loops,
            'while_loops': while_loops,
            'total_loops': for_loops + while_loops,
            'if_statements': if_statements,
            'nested_depth': nested_depth,
            'recursive_calls': recursive_calls,
            'array_operations': code.count('['),
            'pointer_operations': code.count('*'),
            'malloc_calls': tokens.count('malloc'),
            'identifier_count': tokens.count('IDENTIFIER'),
            'number_count': tokens.count('NUMBER'),
            'operator_density': len([t for t in tokens if t in ['+', '-', '*', '/', '%']]) / max(len(tokens), 1),
            'recursion_depth': recursion_depth,
            'nested_for_loops': nested_for_count,
            'exponential_recursion': 1 if is_exponential_recursion else 0,
            'branch_count': branch_count,
            'halving_pattern': halving_pattern,
            'exponential_indicators': exponential_indicators,
            'triple_nested_loops': triple_nested,
            'recursive_call_ratio': recursive_call_ratio,
        }
        
        return features
    
    def _count_triple_nested_loops(self, code):
        """Detecta si hay 3 o más niveles de bucles anidados"""
        max_depth = 0
        current_depth = 0
        
        for char in code:
            if char == '{':
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif char == '}':
                current_depth = max(0, current_depth - 1)
        
        # Retornar 1 si hay triple nesting, 0 si no
        return 1 if max_depth >= 3 else 0
    
    def _measure_recursion_depth(self, code):
        """Mide cuán profunda es la recursión (nivel de anidamiento)"""
        # Contar paréntesis para medir profundidad de llamadas
        max_paren_depth = 0
        current_depth = 0
        for char in code:
            if char == '(':
                current_depth += 1
                max_paren_depth = max(max_paren_depth, current_depth)
            elif char == ')':
                current_depth = max(0, current_depth - 1)
        return max_paren_depth
    
    def _count_nested_for_loops(self, code):
        """Cuenta cuántos niveles de for anidados hay (más preciso)"""
        max_nested_depth = 0
        
        # Buscar patrones de for anidados
        lines = code.split('\n')
        current_depth = 0
        
        for line in lines:
            # Contar 'for' con '{' inmediatamente
            for_with_brace = len(re.findall(r'for\s*\([^)]*\)\s*\{', line))
            if for_with_brace > 0:
                current_depth += for_with_brace
                max_nested_depth = max(max_nested_depth, current_depth)
            
            # Contar cierres de braces
            braces = line.count('}')
            current_depth = max(0, current_depth - braces)
        
        return max_nested_depth
    
    def _count_exact_nested_loop_count(self, code):
        """Cuenta la profundidad exacta de for loops anidados (contador de loops)"""
        # Contar cuántos bucles for hay y su profundidad máxima
        # Devuelve el NÚMERO EXACTO de niveles de anidamiento de for
        lines = code.split('\n')
        max_nesting = 0
        current_nesting = 0
        
        for line in lines:
            # Contar aperturas de 'for'
            for_count = len(re.findall(r'for\s*\(', line))
            # Contar aperturas de 'while'
            while_count = len(re.findall(r'while\s*\(', line))
            
            # Aumentar profundidad
            current_nesting += for_count + while_count
            max_nesting = max(max_nesting, current_nesting)
            
            # Contar cierres de braces para disminuir profundidad
            open_braces = line.count('{')
            close_braces = line.count('}')
            
            # Ajustar profundidad (considerar solo braces de loops)
            net_braces = open_braces - close_braces
            if net_braces < 0:
                current_nesting = max(0, current_nesting + net_braces)
        
        return max_nesting
    
    def _has_multiple_recursive_calls_in_function(self, code):
        """Detecta si una función se llama a sí misma múltiples veces (O(2^n))"""
        # Patrón: función llama a sí misma múltiples veces en el mismo statement
        # Ej: return fibonacci(n-1) + fibonacci(n-2)
        
        # Extraer función principal (la que está siendo definida)
        func_match = re.search(r'\b(\w+)\s*\([^)]*\)\s*\{', code)
        if not func_match:
            return False
        
        func_name = func_match.group(1)
        
        # Buscar si se llama a sí misma múltiples veces en una sola línea
        lines = code.split('\n')
        for line in lines:
            # Contar cuántas veces se llama a sí misma en esta línea
            call_count = line.count(f'{func_name}(')
            if call_count >= 2:  # Mínimo 2 llamadas en la misma línea
                return True
        
        return False

    
    def calculate_nesting_depth(self, code):
        """Calcula la profundidad máxima de anidamiento de ciclos/condicionales"""
        max_depth = 0
        current_depth = 0
        
        for char in code:
            if char == '{':
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif char == '}':
                current_depth = max(0, current_depth - 1)
        
        return max_depth
    
    def is_c_code(self, code):
        """Verifica si el código es C válido"""
        code_lower = code.lower()
        
        # Detectar lenguajes no-C (con orden de especificidad)
        # Java primero (más específico)
        java_indicators = ['public class', 'public static', 'system.out', 'new ', 'throws', '@override', 'extends', 'implements', 'interface ']
        java_count = sum(1 for ind in java_indicators if ind in code_lower)
        
        # C++ (antes de Python para evitar conflictos)
        cpp_indicators = ['std::', '#include <iostream>', 'cout', 'namespace std', 'template', 'vector<', 'string', '::']
        cpp_count = sum(1 for ind in cpp_indicators if ind in code_lower)
        
        # JavaScript
        js_indicators = ['function ', 'const ', 'let ', 'var ', '=>', 'console.log', 'document.', 'window.']
        js_count = sum(1 for ind in js_indicators if ind in code_lower)
        
        # Python (menos estricto para evitar falsos positivos)
        python_indicators = ['print(', 'def ', 'import ', 'class ', ':\n', '__', 'self.', 'self ', 'elif ', 'except:']
        python_count = sum(1 for ind in python_indicators if ind in code_lower)
        
        if java_count >= 2:
            return False, "Java"
        if cpp_count >= 2:
            return False, "C++"
        if js_count >= 2:
            return False, "JavaScript"
        if python_count >= 2:
            return False, "Python"
        
        # Verificar que tenga características de C
        c_indicators = ['int ', 'void ', 'char ', 'float ', 'for ', 'while ', 'if ', '#include', 'main']
        c_count = sum(1 for ind in c_indicators if ind in code_lower)
        
        if c_count >= 1 or ('#include' in code or 'main' in code or any(x in code for x in ['int ', 'void ', 'for ', 'while '])):
            return True, "C"
        
        return None, None  # Código ambiguo
    
    def save(self, filename="tokenizer.pkl"):
        """Guarda el tokenizador"""
        import sys
        # Asegurar que la clase esté disponible en __main__
        sys.modules['__main__'].CodeTokenizer = CodeTokenizer
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(filename="tokenizer.pkl"):
        """Carga el tokenizador"""
        import sys
        sys.modules['__main__'].CodeTokenizer = CodeTokenizer
        with open(filename, 'rb') as f:
            return pickle.load(f)


class ComplexityAnalyzerNN:
    """Red Neuronal para análisis de complejidad"""
    
    def __init__(self):
        self.num_classes = 7
        self.model = None
        self.scaler = StandardScaler()
        self.tokenizer = CodeTokenizer()
        self.complexity_labels = ["O(1)", "O(log n)", "O(n)", "O(n log n)", "O(n²)", "O(n³)", "O(2^n)"]
    
    def build_model(self, epochs=500):
        """Construye el modelo de red neuronal"""
        model = MLPClassifier(
            hidden_layer_sizes=(256, 128, 64, 32),
            activation='relu',
            solver='adam',
            learning_rate='adaptive',
            learning_rate_init=0.001,
            batch_size=16,
            max_iter=epochs,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.2,
            n_iter_no_change=50,
            verbose=True
        )
        self.model = model
        return model
    
    def prepare_data(self, dataset_file="algorithms_dataset.json"):
        """Prepara datos para entrenamiento"""
        with open(dataset_file, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        X = []
        y = []
        
        print("Extrayendo características...")
        for i, item in enumerate(dataset):
            if (i + 1) % max(1, len(dataset) // 10) == 0:
                print(f"  {i + 1}/{len(dataset)}")
            
            code = item['code']
            label = item['label']
            
            features = self.tokenizer.extract_features(code)
            feature_vector = [
                features['token_count'],
                features['for_loops'],
                features['while_loops'],
                features['total_loops'],
                features['if_statements'],
                features['nested_depth'],
                features['recursive_calls'],
                features['array_operations'],
                features['pointer_operations'],
                features['malloc_calls'],
                features['identifier_count'],
                features['number_count'],
                features['operator_density'],
                features['recursion_depth'],
                features['nested_for_loops'],
                features['exponential_recursion'],
                features['branch_count'],
                features['halving_pattern'],
                features['exponential_indicators'],
                features['triple_nested_loops'],
                features['recursive_call_ratio'],
            ]
            
            X.append(feature_vector)
            y.append(label)
        
        X = np.array(X)
        y = np.array(y)
        
        return X, y
    
    def train(self, dataset_file="algorithms_dataset.json", epochs=500, model_save_path="complexity_model.pkl"):
        """Entrena el modelo"""
        print("\n" + "="*60)
        print("Preparando datos...")
        print("="*60)
        X, y = self.prepare_data(dataset_file)
        
        print(f"\nDataset shape: {X.shape}")
        print(f"Labels shape: {y.shape}")
        print(f"Distribución de clases: {np.bincount(y)}")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\nTrain set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        
        print("\nNormalizando datos...")
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        print("\n" + "="*60)
        print("Construyendo y entrenando modelo...")
        print("="*60)
        self.build_model(epochs)
        
        print(f"\nArquitectura: MLPClassifier")
        print(f"Capas ocultas: (256, 128, 64, 32)")
        print(f"Activación: ReLU")
        print(f"Optimizer: Adam")
        print(f"Iteraciones máximas: {epochs}")
        
        print(f"\nEntrenando...")
        self.model.fit(X_train, y_train)
        
        print("\n" + "="*60)
        print("Evaluación del modelo")
        print("="*60)
        
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        print(f"\nAccuracy en entrenamiento: {train_score:.4f}")
        print(f"Accuracy en test: {test_score:.4f}")
        
        y_pred = self.model.predict(X_test)
        
        print("\nReporte de clasificación:")
        unique_labels = np.unique(y_test)
        target_names = [self.complexity_labels[i] for i in unique_labels]
        print(classification_report(y_test, y_pred, labels=unique_labels, target_names=target_names))
        
        print("\nMatriz de confusión:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        
        joblib.dump(self.model, model_save_path)
        joblib.dump(self.scaler, "scaler.pkl")
        self.tokenizer.save("tokenizer.pkl")
        
        print(f"\n[OK] Modelo guardado en: {model_save_path}")
        print(f"[OK] Scaler guardado en: scaler.pkl")
        print(f"[OK] Tokenizador guardado en: tokenizer.pkl")
        
        return {"train_score": train_score, "test_score": test_score}
    
    def predict_complexity(self, code):
        """Predice la complejidad de un código C"""
        if self.model is None:
            raise ValueError("Modelo no entrenado. Cargue un modelo primero.")
        
        # Validar que sea código C
        is_c, language = self.tokenizer.is_c_code(code)
        
        if is_c is False:
            raise ValueError(f"Error: El código proporcionado es {language}, no C. Por favor, ingrese código C válido.")
        
        if is_c is None:
            raise ValueError("Error: No se puede determinar si el código es C. Asegúrese de que sea código C válido con estructuras como 'int', 'void', 'for', 'while', etc.")
        
        features = self.tokenizer.extract_features(code)
        feature_vector = np.array([
            features['token_count'],
            features['for_loops'],
            features['while_loops'],
            features['total_loops'],
            features['if_statements'],
            features['nested_depth'],
            features['recursive_calls'],
            features['array_operations'],
            features['pointer_operations'],
            features['malloc_calls'],
            features['identifier_count'],
            features['number_count'],
            features['operator_density'],
            features['recursion_depth'],
            features['nested_for_loops'],
            features['exponential_recursion'],
            features['branch_count'],
            features['halving_pattern'],
            features['exponential_indicators'],
            features['triple_nested_loops'],
            features['recursive_call_ratio'],
        ]).reshape(1, -1)
        
        feature_vector = self.scaler.transform(feature_vector)
        predicted_idx = self.model.predict(feature_vector)[0]
        
        try:
            probabilities = self.model.predict_proba(feature_vector)[0]
            confidence = probabilities[predicted_idx]
            prob_dict = {self.complexity_labels[i]: float(probabilities[i]) for i in range(len(self.complexity_labels))}
        except AttributeError:
            confidence = 1.0
            prob_dict = {self.complexity_labels[i]: 0.0 for i in range(len(self.complexity_labels))}
            prob_dict[self.complexity_labels[predicted_idx]] = 1.0
        
        tokens = self.tokenizer.tokenize_code(code)
        
        return {
            'complexity': self.complexity_labels[predicted_idx],
            'confidence': float(confidence),
            'probabilities': prob_dict,
            'tokens': tokens[:50],
            'features': features
        }
    
    def load_model(self, model_path="complexity_model.pkl", tokenizer_path="tokenizer.pkl", scaler_path="scaler.pkl"):
        """Carga un modelo entrenado"""
        self.model = joblib.load(model_path)
        self.tokenizer = CodeTokenizer.load(tokenizer_path)
        self.scaler = joblib.load(scaler_path)
        print(f"[OK] Modelo cargado desde: {model_path}")
        print(f"[OK] Tokenizador cargado desde: {tokenizer_path}")
        print(f"[OK] Scaler cargado desde: {scaler_path}")


if __name__ == "__main__":
    analyzer = ComplexityAnalyzerNN()
    history = analyzer.train(epochs=500)
