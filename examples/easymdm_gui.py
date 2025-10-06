#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interface graphique EasyMDM Advanced
Traduction française et interface GUI pour EasyMDM Advanced
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
import os
import pandas as pd
import logging
import threading
from pathlib import Path
import warnings
from numba.core.errors import NumbaWarning

# Suppress warnings
warnings.simplefilter('ignore', category=NumbaWarning)

# Mock imports pour la démonstration (remplacer par les vrais imports)
class MDMEngine:
    def __init__(self, config):
        self.config = config
    
    def process(self):
        import time
        time.sleep(2)  # Simulation du traitement
        return MockResult()

class MockResult:
    def __init__(self):
        self.golden_records = pd.DataFrame({
            'first_name': ['John', 'Jane'],
            'last_name': ['Smith', 'Doe'],
            'address': ['123 Main St', '456 Oak Ave'],
            'similar_record_ids': ['[1,2]', '[3,4]'],
            'logic': ['Source priority', 'Most recent']
        })
        self.execution_time = 2.5
        self.output_files = ['output1.csv', 'output2.csv']
        self.processing_stats = {'memory_usage': '256MB', 'candidate_pairs': 1500}

class MDMConfig:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def save_yaml(self, filename):
        with open(filename, 'w') as f:
            f.write(f"# Configuration EasyMDM\n# Généré automatiquement\n")

class DatabaseConfig:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

class BlockingConfig:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

class SimilarityConfig:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

class ThresholdConfig:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

class SurvivorshipRule:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

class PriorityCondition:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class EasyMDMGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("EasyMDM Advanced - Interface Graphique")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        
        # Variables
        self.csv_file_path = tk.StringVar()
        self.output_path = tk.StringVar(value="./sortie_mdm")
        self.processing_running = False
        
        self.setup_gui()
        
    def setup_gui(self):
        """Configuration de l'interface graphique"""
        
        # Style
        style = ttk.Style()
        style.theme_use('clam')
        
        # Frame principal
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configuration du grid
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # ===== SECTION 1: SÉLECTION DE FICHIER =====
        file_frame = ttk.LabelFrame(main_frame, text="📁 Sélection du fichier de données", padding="10")
        file_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N), pady=5)
        file_frame.columnconfigure(1, weight=1)
        
        ttk.Label(file_frame, text="Fichier CSV:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        ttk.Entry(file_frame, textvariable=self.csv_file_path, width=50).grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 10))
        ttk.Button(file_frame, text="Parcourir", command=self.browse_csv_file).grid(row=0, column=2, sticky=tk.W)
        ttk.Button(file_frame, text="Créer échantillon", command=self.create_sample_data).grid(row=0, column=3, sticky=tk.W, padx=(10, 0))
        
        # ===== SECTION 2: CONFIGURATION =====
        config_frame = ttk.LabelFrame(main_frame, text="⚙️ Configuration MDM", padding="10")
        config_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N), pady=5)
        config_frame.columnconfigure(1, weight=1)
        
        # Configuration du blocking
        ttk.Label(config_frame, text="Colonnes de blocage:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.blocking_columns = tk.StringVar(value="first_name,last_name")
        ttk.Entry(config_frame, textvariable=self.blocking_columns, width=40).grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 10))
        
        ttk.Label(config_frame, text="Méthode de blocage:").grid(row=1, column=0, sticky=tk.W, padx=(0, 10), pady=(5, 0))
        self.blocking_method = ttk.Combobox(config_frame, values=["exact", "fuzzy"], state="readonly", width=20)
        self.blocking_method.set("exact")
        self.blocking_method.grid(row=1, column=1, sticky=tk.W, pady=(5, 0))
        
        # Seuils
        ttk.Label(config_frame, text="Seuil de révision:").grid(row=2, column=0, sticky=tk.W, padx=(0, 10), pady=(5, 0))
        self.review_threshold = tk.DoubleVar(value=0.7)
        ttk.Scale(config_frame, from_=0.0, to=1.0, variable=self.review_threshold, orient=tk.HORIZONTAL, length=200).grid(row=2, column=1, sticky=(tk.W, tk.E), pady=(5, 0))
        
        ttk.Label(config_frame, text="Seuil de fusion auto:").grid(row=3, column=0, sticky=tk.W, padx=(0, 10), pady=(5, 0))
        self.auto_merge_threshold = tk.DoubleVar(value=0.85)
        ttk.Scale(config_frame, from_=0.0, to=1.0, variable=self.auto_merge_threshold, orient=tk.HORIZONTAL, length=200).grid(row=3, column=1, sticky=(tk.W, tk.E), pady=(5, 0))
        
        # Chemin de sortie
        ttk.Label(config_frame, text="Dossier de sortie:").grid(row=4, column=0, sticky=tk.W, padx=(0, 10), pady=(5, 0))
        ttk.Entry(config_frame, textvariable=self.output_path, width=40).grid(row=4, column=1, sticky=(tk.W, tk.E), padx=(0, 10), pady=(5, 0))
        ttk.Button(config_frame, text="Parcourir", command=self.browse_output_folder).grid(row=4, column=2, sticky=tk.W, pady=(5, 0))
        
        # ===== SECTION 3: EXEMPLES PRÉDÉFINIS =====
        examples_frame = ttk.LabelFrame(main_frame, text="🔧 Exemples prédéfinis", padding="10")
        examples_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N), pady=5)
        
        button_frame = ttk.Frame(examples_frame)
        button_frame.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        ttk.Button(button_frame, text="1. Traitement CSV basique", 
                  command=lambda: self.run_example(1)).grid(row=0, column=0, padx=5, pady=2)
        ttk.Button(button_frame, text="2. Similarité avancée", 
                  command=lambda: self.run_example(2)).grid(row=0, column=1, padx=5, pady=2)
        ttk.Button(button_frame, text="3. Config PostgreSQL", 
                  command=lambda: self.run_example(3)).grid(row=0, column=2, padx=5, pady=2)
        ttk.Button(button_frame, text="4. Optimisation performance", 
                  command=lambda: self.run_example(4)).grid(row=1, column=0, padx=5, pady=2)
        ttk.Button(button_frame, text="5. Survivance personnalisée", 
                  command=lambda: self.run_example(5)).grid(row=1, column=1, padx=5, pady=2)
        
        # ===== SECTION 4: CONTRÔLES =====
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        
        self.run_button = ttk.Button(control_frame, text="🚀 Lancer le traitement", 
                                    command=self.run_custom_processing, style="Accent.TButton")
        self.run_button.grid(row=0, column=0, padx=5)
        
        self.stop_button = ttk.Button(control_frame, text="⏹️ Arrêter", 
                                     command=self.stop_processing, state="disabled")
        self.stop_button.grid(row=0, column=1, padx=5)
        
        ttk.Button(control_frame, text="📋 Effacer log", 
                  command=self.clear_log).grid(row=0, column=2, padx=5)
        
        # Barre de progression
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(control_frame, variable=self.progress_var, 
                                          maximum=100, length=300, mode='determinate')
        self.progress_bar.grid(row=0, column=3, padx=10, sticky=(tk.W, tk.E))
        control_frame.columnconfigure(3, weight=1)
        
        # ===== SECTION 5: LOG =====
        log_frame = ttk.LabelFrame(main_frame, text="📊 Journal d'exécution", padding="10")
        log_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        main_frame.rowconfigure(4, weight=1)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=15, width=80, 
                                                 font=('Consolas', 10), wrap=tk.WORD)
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configuration du logging
        self.setup_logging()
        
        # Message de bienvenue
        self.log_message("✅ Interface EasyMDM Advanced initialisée")
        self.log_message("💡 Sélectionnez un fichier CSV ou créez un échantillon pour commencer")
        
    def setup_logging(self):
        """Configuration du système de logging"""
        
        class GUILogHandler(logging.Handler):
            def __init__(self, text_widget):
                super().__init__()
                self.text_widget = text_widget
                
            def emit(self, record):
                msg = self.format(record)
                # Thread-safe update
                self.text_widget.after(0, lambda: self.text_widget.insert(tk.END, msg + '\n'))
                self.text_widget.after(0, lambda: self.text_widget.see(tk.END))
        
        self.logger = logging.getLogger('EasyMDM_GUI')
        self.logger.setLevel(logging.INFO)
        
        # Supprimer les handlers existants
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
            
        # Ajouter le handler GUI
        gui_handler = GUILogHandler(self.log_text)
        gui_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s', 
                                                  datefmt='%H:%M:%S'))
        self.logger.addHandler(gui_handler)
        
    def log_message(self, message):
        """Ajouter un message au log"""
        timestamp = pd.Timestamp.now().strftime('%H:%M:%S')
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()
        
    def browse_csv_file(self):
        """Parcourir et sélectionner un fichier CSV"""
        filename = filedialog.askopenfilename(
            title="Sélectionner un fichier CSV",
            filetypes=[("Fichiers CSV", "*.csv"), ("Tous les fichiers", "*.*")]
        )
        if filename:
            self.csv_file_path.set(filename)
            self.log_message(f"📁 Fichier sélectionné: {os.path.basename(filename)}")
            
    def browse_output_folder(self):
        """Parcourir et sélectionner un dossier de sortie"""
        folder = filedialog.askdirectory(title="Sélectionner le dossier de sortie")
        if folder:
            self.output_path.set(folder)
            self.log_message(f"📂 Dossier de sortie: {folder}")
            
    def create_sample_data(self):
        """Créer des données d'échantillon"""
        sample_data = {
            'first_name': ['Jean', 'Jon', 'Marie', 'Jean', 'Mariette'],
            'last_name': ['Dupont', 'Dupont', 'Martin', 'Dupond', 'Martin'],
            'address': ['123 Rue Principale', '123 Rue Principale', '456 Avenue des Chênes', 
                       '123 Rue Principale Apt 2', '456 Avenue des Chênes'],
            'city': ['Paris', 'Paris', 'Lyon', 'Paris', 'Lyon'],
            'phone': ['01-23-45-67-89', '01.23.45.67.89', '04-56-78-90-12', 
                     '(01) 23 45 67 89', '04.56.78.90.12'],
            'email': ['jean@email.com', 'jean@email.com', 'marie@travail.com', 
                     'jean.dupont@email.com', 'marie.martin@email.com'],
            'source': ['CRM', 'Import', 'Manuel', 'CRM', 'Manuel'],
            'last_updated': ['2023-01-15', '2023-01-10', '2023-02-20', '2023-01-20', '2023-02-25']
        }
        
        df = pd.DataFrame(sample_data)
        filename = 'echantillon_clients.csv'
        df.to_csv(filename, index=False, encoding='utf-8')
        
        self.csv_file_path.set(os.path.abspath(filename))
        self.log_message(f"✅ Données d'échantillon créées: {filename}")
        self.log_message(f"📊 {len(df)} enregistrements générés")
        
    def clear_log(self):
        """Effacer le journal"""
        self.log_text.delete(1.0, tk.END)
        self.log_message("📋 Journal effacé")
        
    def update_progress(self, value):
        """Mettre à jour la barre de progression"""
        self.progress_var.set(value)
        self.root.update_idletasks()
        
    def set_processing_state(self, running):
        """Changer l'état des contrôles pendant le traitement"""
        self.processing_running = running
        state = "disabled" if running else "normal"
        stop_state = "normal" if running else "disabled"
        
        self.run_button.configure(state=state)
        self.stop_button.configure(state=stop_state)
        
        if running:
            self.progress_bar.configure(mode='indeterminate')
            self.progress_bar.start()
        else:
            self.progress_bar.stop()
            self.progress_bar.configure(mode='determinate')
            self.update_progress(0)
            
    def stop_processing(self):
        """Arrêter le traitement en cours"""
        self.processing_running = False
        self.log_message("⏹️ Arrêt demandé par l'utilisateur")
        
    def run_custom_processing(self):
        """Lancer le traitement personnalisé"""
        if not self.csv_file_path.get():
            messagebox.showerror("Erreur", "Veuillez sélectionner un fichier CSV")
            return
            
        if not os.path.exists(self.csv_file_path.get()):
            messagebox.showerror("Erreur", "Le fichier CSV sélectionné n'existe pas")
            return
            
        # Lancer dans un thread séparé
        thread = threading.Thread(target=self._run_custom_processing_thread)
        thread.daemon = True
        thread.start()
        
    def _run_custom_processing_thread(self):
        """Thread de traitement personnalisé"""
        try:
            self.set_processing_state(True)
            self.log_message("🚀 Démarrage du traitement personnalisé...")
            
            # Création de la configuration
            config = MDMConfig(
                source=DatabaseConfig(
                    type='csv',
                    file_path=self.csv_file_path.get()
                ),
                blocking=BlockingConfig(
                    columns=self.blocking_columns.get().split(','),
                    method=self.blocking_method.get()
                ),
                similarity=[
                    SimilarityConfig(column='first_name', method='jarowinkler', weight=2.0),
                    SimilarityConfig(column='last_name', method='jarowinkler', weight=2.0),
                    SimilarityConfig(column='address', method='levenshtein', weight=1.5),
                ],
                thresholds=ThresholdConfig(
                    review=self.review_threshold.get(),
                    auto_merge=self.auto_merge_threshold.get(),
                    definite_no_match=0.3
                ),
                survivorship_rules=[
                    SurvivorshipRule(column='last_updated', strategy='most_recent'),
                ],
                output_path=self.output_path.get()
            )
            
            self.log_message("⚙️ Configuration créée")
            self.update_progress(25)
            
            if self.processing_running:
                # Lancer le moteur MDM
                engine = MDMEngine(config)
                self.log_message("🔧 Moteur MDM initialisé")
                self.update_progress(50)
                
                if self.processing_running:
                    result = engine.process()
                    self.update_progress(75)
                    
                    if self.processing_running:
                        # Afficher les résultats
                        self.log_message("✅ Traitement terminé!")
                        self.log_message(f"📊 Enregistrements dorés: {len(result.golden_records)}")
                        self.log_message(f"⏱️ Temps d'exécution: {result.execution_time:.2f} secondes")
                        self.log_message(f"📁 Fichiers de sortie: {len(result.output_files)}")
                        
                        self.update_progress(100)
                        
                        # Afficher un aperçu des résultats
                        if not result.golden_records.empty:
                            self.log_message("\n🏆 Aperçu des enregistrements dorés:")
                            for idx, row in result.golden_records.head(3).iterrows():
                                self.log_message(f"   • {row.get('first_name', 'N/A')} {row.get('last_name', 'N/A')} - {row.get('logic', 'N/A')}")
                    
        except Exception as e:
            self.log_message(f"❌ Erreur durant le traitement: {str(e)}")
            messagebox.showerror("Erreur", f"Erreur durant le traitement:\n{str(e)}")
            
        finally:
            self.set_processing_state(False)
            
    def run_example(self, example_number):
        """Lancer un exemple prédéfini"""
        thread = threading.Thread(target=self._run_example_thread, args=(example_number,))
        thread.daemon = True
        thread.start()
        
    def _run_example_thread(self, example_number):
        """Thread pour les exemples"""
        try:
            self.set_processing_state(True)
            
            examples = {
                1: self._example_1_csv_basic,
                2: self._example_2_advanced_similarity,
                3: self._example_3_postgresql_config,
                4: self._example_4_performance,
                5: self._example_5_survivorship
            }
            
            if example_number in examples:
                examples[example_number]()
            else:
                self.log_message(f"❌ Exemple {example_number} non trouvé")
                
        except Exception as e:
            self.log_message(f"❌ Erreur dans l'exemple {example_number}: {str(e)}")
            
        finally:
            self.set_processing_state(False)
            
    def _example_1_csv_basic(self):
        """Exemple 1: Traitement CSV basique"""
        self.log_message("=" * 60)
        self.log_message("EXEMPLE 1: Traitement CSV basique")
        self.log_message("=" * 60)
        
        # Créer des données d'exemple
        sample_data = {
            'first_name': ['Jean', 'Jon', 'Marie', 'Jean', 'Mariette'],
            'last_name': ['Dupont', 'Dupont', 'Martin', 'Dupond', 'Martin'],
            'address': ['123 Rue Principale', '123 Rue Principale', '456 Avenue des Chênes', 
                       '123 Rue Principale', '456 Avenue des Chênes'],
            'city': ['Paris', 'Paris', 'Lyon', 'Paris', 'Lyon'],
            'phone': ['01-23-45-67-89', '01-23-45-67-89', '04-56-78-90-12', 
                     '(01) 23 45 67 89', '04.56.78.90.12'],
            'email': ['jean@email.com', 'jean@email.com', 'marie@travail.com', 
                     'jean.dupont@email.com', 'marie.martin@email.com'],
            'source': ['CRM', 'Import', 'Manuel', 'CRM', 'Manuel'],
            'last_updated': ['2023-01-15', '2023-01-10', '2023-02-20', '2023-01-20', '2023-02-25']
        }
        
        df = pd.DataFrame(sample_data)
        csv_file = 'exemple_clients.csv'
        df.to_csv(csv_file, index=False, encoding='utf-8')
        self.log_message(f"📁 Fichier d'exemple créé: {csv_file}")
        
        self.update_progress(30)
        
        # Configuration
        config = MDMConfig(
            source=DatabaseConfig(type='csv', file_path=csv_file),
            blocking=BlockingConfig(columns=['first_name', 'last_name'], method='exact'),
            similarity=[
                SimilarityConfig(column='first_name', method='jarowinkler', weight=2.0),
                SimilarityConfig(column='last_name', method='jarowinkler', weight=2.0),
                SimilarityConfig(column='address', method='levenshtein', weight=1.5),
            ],
            thresholds=ThresholdConfig(review=0.7, auto_merge=0.85, definite_no_match=0.3),
            survivorship_rules=[
                SurvivorshipRule(column='last_updated', strategy='most_recent'),
            ],
            output_path='./sortie_exemple1'
        )
        
        self.update_progress(60)
        
        # Traitement
        engine = MDMEngine(config)
        result = engine.process()
        
        self.update_progress(90)
        
        # Résultats
        self.log_message("✅ Traitement terminé!")
        self.log_message(f"   Enregistrements d'entrée: {len(df)}")
        self.log_message(f"   Enregistrements dorés: {len(result.golden_records)}")
        self.log_message(f"   Temps de traitement: {result.execution_time:.2f} secondes")
        
        if not result.golden_records.empty:
            self.log_message("\n📊 Enregistrements dorés:")
            for idx, row in result.golden_records.iterrows():
                self.log_message(f"   • {row.get('first_name', 'N/A')} {row.get('last_name', 'N/A')} - {row.get('logic', 'N/A')}")
        
        self.update_progress(100)
        
        # Nettoyage
        if os.path.exists(csv_file):
            os.remove(csv_file)
            
    def _example_2_advanced_similarity(self):
        """Exemple 2: Configuration de similarité avancée"""
        self.log_message("=" * 60)
        self.log_message("EXEMPLE 2: Configuration de similarité avancée")
        self.log_message("=" * 60)
        
        # Test de méthodes de similarité
        test_pairs = [
            ("Jean Dupont", "Jon Dupont"),
            ("123 Rue Principale", "123 Rue Princ."),
            ("jean@email.com", "jean@gmail.com"),
            ("Paris", "PARIS"),
            ("01-23-45-67-89", "(01) 23 45 67 89")
        ]
        
        methods = ['jarowinkler', 'levenshtein', 'exact']
        
        self.log_message("🔍 Comparaison des méthodes de similarité:")
        self.log_message("-" * 50)
        
        for str1, str2 in test_pairs:
            self.log_message(f"\nComparaison: '{str1}' vs '{str2}'")
            for method in methods:
                # Simulation du calcul de similarité
                import random
                score = random.uniform(0.6, 1.0) if str1.lower() in str2.lower() else random.uniform(0.0, 0.5)
                self.log_message(f"  {method:15}: {score:.3f}")
        
        self.log_message("\n💡 Recommandations de méthodes:")
        self.log_message("-" * 40)
        self.log_message("name           : jarowinkler, exact")
        self.log_message("address        : levenshtein, cosine")
        self.log_message("phone          : levenshtein")
        self.log_message("email          : exact, jarowinkler")
        
        self.update_progress(100)
        
    def _example_3_postgresql_config(self):
        """Exemple 3: Configuration PostgreSQL"""
        self.log_message("=" * 60)
        self.log_message("EXEMPLE 3: Configuration PostgreSQL avancée")
        self.log_message("=" * 60)
        
        config = MDMConfig(
            source=DatabaseConfig(
                type='postgresql',
                host='localhost',
                port=5432,
                database='customer_db',
                username='mdm_user',
                password='***',
                schema='public',
                table='customers'
            ),
            blocking=BlockingConfig(
                columns=['first_name', 'last_name'],
                method='fuzzy',
                threshold=0.8
            ),
            similarity=[
                SimilarityConfig(column='first_name', method='jarowinkler', weight=2.0),
                SimilarityConfig(column='last_name', method='jarowinkler', weight=2.0),
                SimilarityConfig(column='address', method='cosine', weight=1.5),
            ],
            thresholds=ThresholdConfig(review=0.75, auto_merge=0.9, definite_no_match=0.3),
            survivorship_rules=[
                SurvivorshipRule(column='updated_at', strategy='most_recent'),
                SurvivorshipRule(column='data_source', strategy='source_priority',
                               source_order=['MASTER_SYSTEM', 'CRM', 'IMPORT', 'MANUAL']),
            ],
            output_path='./sortie_postgresql'
        )
        
        # Sauvegarder la configuration
        config_file = 'config_postgresql.yaml'
        config.save_yaml(config_file)
        self.log_message(f"✅ Configuration PostgreSQL créée: {config_file}")
        
        self.log_message("\n🔍 Validation de la configuration:")
        self.log_message(f"   Type de source: {config.source.type}")
        self.log_message(f"   Méthode de blocage: {config.blocking.method}")
        self.log_message(f"   Configurations de similarité: {len(config.similarity)}")
        self.log_message(f"   Règles de survivance: {len(config.survivorship_rules)}")
        
        self.log_message("\n📊 Connecteurs de base de données disponibles:")
        self.log_message("   • PostgreSQL, MySQL, SQLite, CSV, Excel")
        
        self.update_progress(100)
        
        # Nettoyage
        if os.path.exists(config_file):
            os.remove(config_file)
            
    def _example_4_performance(self):
        """Exemple 4: Optimisation des performances"""
        self.log_message("=" * 60)
        self.log_message("EXEMPLE 4: Optimisation des performances")
        self.log_message("=" * 60)
        
        # Génération de données volumineuses (simulation)
        import random
        import time
        
        self.log_message("📊 Génération d'un jeu de données volumineux...")
        self.update_progress(20)
        
        # Simulation de la génération
        time.sleep(1)
        record_count = 10000
        self.log_message(f"📊 Jeu de données généré: {record_count:,} enregistrements")
        
        self.update_progress(40)
        
        # Configuration optimisée
        self.log_message("⚙️ Configuration optimisée pour les performances:")
        self.log_message("   • Méthode de blocage: exact (plus rapide)")
        self.log_message("   • Taille de lot: 5000")
        self.log_message("   • Parallélisation: 4 processus")
        self.log_message("   • Utilisation de chunks: activée")
        
        self.update_progress(60)
        
        # Simulation du traitement
        self.log_message("🚀 Démarrage du traitement optimisé...")
        time.sleep(2)  # Simulation
        
        self.update_progress(90)
        
        # Résultats de performance simulés
        processing_time = 15.7
        golden_records = 8520
        
        self.log_message("⚡ Résultats de performance:")
        self.log_message(f"   Enregistrements d'entrée: {record_count:,}")
        self.log_message(f"   Enregistrements dorés: {golden_records:,}")
        self.log_message(f"   Temps de traitement: {processing_time:.2f} secondes")
        self.log_message(f"   Enregistrements/seconde: {record_count/processing_time:.1f}")
        self.log_message(f"   Utilisation mémoire: 512MB")
        self.log_message(f"   Paires candidates: 45,230")
        
        self.update_progress(100)
        
    def _example_5_survivorship(self):
        """Exemple 5: Stratégies de survivance personnalisées"""
        self.log_message("=" * 60)
        self.log_message("EXEMPLE 5: Stratégies de survivance personnalisées")
        self.log_message("=" * 60)
        
        # Données d'exemple avec indicateurs de qualité
        sample_data = {
            'customer_id': [1, 2, 3, 4, 5],
            'first_name': ['Jean', 'Jean', 'Marie', 'Marie', 'Robert'],
            'last_name': ['Dupont', 'Dupont', 'Martin', 'Martin', 'Johnson'],
            'email': ['jean@email.com', 'j.dupont@entreprise.com', 'marie@travail.com', 
                     'marie.martin@email.com', 'rob@email.com'],
            'data_source': ['CRM', 'ERP', 'Manuel', 'Import', 'CRM'],
            'quality_score': [95, 85, 90, 75, 98],
            'is_verified': [True, False, True, False, True],
            'last_updated': ['2023-03-15', '2023-02-10', '2023-03-20', '2023-01-05', '2023-03-25']
        }
        
        df = pd.DataFrame(sample_data)
        self.log_message("📊 Données d'exemple pour les tests de survivance:")
        self.log_message(f"   {len(df)} enregistrements avec indicateurs de qualité")
        
        self.update_progress(30)
        
        # Configuration avancée de survivance
        self.log_message("\n⚙️ Configuration de survivance avancée:")
        self.log_message("   • Plus récent: last_updated")
        self.log_message("   • Priorité source: CRM > ERP > Manuel > Import") 
        self.log_message("   • Plus long: address")
        self.log_message("   • Valeur plus élevée: quality_score")
        
        self.update_progress(60)
        
        # Test sur cluster Jean Dupont
        jean_cluster = df[df['first_name'] == 'Jean'].copy()
        self.log_message(f"\n🔍 Test de survivance sur cluster Jean Dupont:")
        self.log_message(f"   {len(jean_cluster)} enregistrements dans le cluster")
        
        # Simulation de la résolution
        self.log_message("\n✅ Résultat de survivance:")
        self.log_message("   ID du survivant: 1")
        self.log_message("   Logique de résolution: Vérification + Score qualité")
        self.log_message("   Score de confiance: 0.92")
        
        self.update_progress(80)
        
        # Enregistrement doré
        self.log_message("\n📋 Champs de l'enregistrement doré:")
        self.log_message("   first_name: Jean")
        self.log_message("   last_name: Dupont")
        self.log_message("   data_source: CRM")
        self.log_message("   quality_score: 95")
        self.log_message("   is_verified: True")
        
        self.update_progress(100)
        
        self.log_message("\n🏆 Résumé final:")
        self.log_message("   Enregistrements d'entrée: 5")
        self.log_message("   Enregistrements dorés: 3")


def main():
    """Fonction principale"""
    root = tk.Tk()
    app = EasyMDMGUI(root)
    
    # Centrer la fenêtre
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f'{width}x{height}+{x}+{y}')
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("\nFermeture de l'application...")
        root.quit()


if __name__ == "__main__":
    main()