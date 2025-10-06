#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EasyMDM Advanced - Interface Web Flask
Application web moderne pour EasyMDM Advanced avec interface utilisateur complète
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, send_file
import os
import pandas as pd
import logging
import threading
import time
import json
from datetime import datetime
from pathlib import Path
import warnings
from numba.core.errors import NumbaWarning
import uuid
from werkzeug.utils import secure_filename

# Suppress warnings
warnings.simplefilter('ignore', category=NumbaWarning)

# Mock imports pour la démonstration (remplacer par les vrais imports)
class MDMEngine:
    def __init__(self, config):
        self.config = config
    
    def process(self):
        time.sleep(3)  # Simulation du traitement
        return MockResult()

class MockResult:
    def __init__(self):
        self.golden_records = pd.DataFrame({
            'first_name': ['Jean', 'Marie', 'Pierre'],
            'last_name': ['Dupont', 'Martin', 'Durand'],
            'address': ['123 Rue Principale', '456 Avenue des Chênes', '789 Boulevard Victor Hugo'],
            'similar_record_ids': ['[1,2]', '[3,4]', '[5]'],
            'logic': ['Priorité source', 'Plus récent', 'Score qualité']
        })
        self.execution_time = 3.2
        self.output_files = ['sortie1.csv', 'sortie2.csv']
        self.processing_stats = {'memory_usage': '256MB', 'candidate_pairs': 1500}

class MDMConfig:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def save_yaml(self, filename):
        with open(filename, 'w', encoding='utf-8') as f:
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

# Configuration Flask
app = Flask(__name__)
app.secret_key = 'easymdm-secret-key-2024'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max

# Variables globales pour le suivi des tâches
processing_tasks = {}
logs_storage = {}

# Configuration des logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Créer les dossiers nécessaires
os.makedirs('uploads', exist_ok=True)
os.makedirs('outputs', exist_ok=True)
os.makedirs('configs', exist_ok=True)

@app.route('/')
def index():
    """Page d'accueil"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Télécharger un fichier CSV"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'message': 'Aucun fichier sélectionné'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'message': 'Aucun fichier sélectionné'})
        
        if file and file.filename.lower().endswith('.csv'):
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{timestamp}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Analyser le fichier CSV
            try:
                df = pd.read_csv(filepath, encoding='utf-8')
                file_info = {
                    'filename': filename,
                    'filepath': filepath,
                    'rows': len(df),
                    'columns': list(df.columns),
                    'size': os.path.getsize(filepath)
                }
                return jsonify({'success': True, 'file_info': file_info})
            except Exception as e:
                return jsonify({'success': False, 'message': f'Erreur lors de la lecture du CSV: {str(e)}'})
        else:
            return jsonify({'success': False, 'message': 'Seuls les fichiers CSV sont acceptés'})
    
    except Exception as e:
        return jsonify({'success': False, 'message': f'Erreur: {str(e)}'})

@app.route('/create_sample')
def create_sample():
    """Créer un fichier d'échantillon"""
    try:
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
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'echantillon_clients_{timestamp}.csv'
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        df.to_csv(filepath, index=False, encoding='utf-8')
        
        file_info = {
            'filename': filename,
            'filepath': filepath,
            'rows': len(df),
            'columns': list(df.columns),
            'size': os.path.getsize(filepath)
        }
        
        return jsonify({'success': True, 'file_info': file_info})
    
    except Exception as e:
        return jsonify({'success': False, 'message': f'Erreur: {str(e)}'})

@app.route('/start_processing', methods=['POST'])
def start_processing():
    """Démarrer le traitement MDM"""
    try:
        data = request.json
        task_id = str(uuid.uuid4())
        
        # Créer la tâche
        task = {
            'id': task_id,
            'status': 'en_cours',
            'progress': 0,
            'start_time': datetime.now(),
            'config': data,
            'logs': []
        }
        
        processing_tasks[task_id] = task
        logs_storage[task_id] = []
        
        # Démarrer le traitement dans un thread séparé
        thread = threading.Thread(target=process_mdm_task, args=(task_id, data))
        thread.daemon = True
        thread.start()
        
        return jsonify({'success': True, 'task_id': task_id})
    
    except Exception as e:
        return jsonify({'success': False, 'message': f'Erreur: {str(e)}'})

@app.route('/run_example/<int:example_id>')
def run_example(example_id):
    """Lancer un exemple prédéfini"""
    try:
        task_id = str(uuid.uuid4())
        
        # Créer la tâche
        task = {
            'id': task_id,
            'status': 'en_cours',
            'progress': 0,
            'start_time': datetime.now(),
            'example_id': example_id,
            'logs': []
        }
        
        processing_tasks[task_id] = task
        logs_storage[task_id] = []
        
        # Démarrer l'exemple dans un thread séparé
        thread = threading.Thread(target=process_example, args=(task_id, example_id))
        thread.daemon = True
        thread.start()
        
        return jsonify({'success': True, 'task_id': task_id})
    
    except Exception as e:
        return jsonify({'success': False, 'message': f'Erreur: {str(e)}'})

@app.route('/task_status/<task_id>')
def task_status(task_id):
    """Obtenir le statut d'une tâche"""
    if task_id in processing_tasks:
        task = processing_tasks[task_id]
        return jsonify({
            'success': True,
            'task': {
                'id': task['id'],
                'status': task['status'],
                'progress': task['progress'],
                'logs': logs_storage.get(task_id, [])[-50:],  # 50 derniers logs
                'start_time': task['start_time'].isoformat() if 'start_time' in task else None,
                'end_time': task.get('end_time', {}).isoformat() if task.get('end_time') else None,
                'result': task.get('result', {})
            }
        })
    else:
        return jsonify({'success': False, 'message': 'Tâche non trouvée'})

@app.route('/stop_task/<task_id>')
def stop_task(task_id):
    """Arrêter une tâche"""
    if task_id in processing_tasks:
        processing_tasks[task_id]['status'] = 'arrete'
        add_log(task_id, '⏹️ Tâche arrêtée par l\'utilisateur')
        return jsonify({'success': True})
    else:
        return jsonify({'success': False, 'message': 'Tâche non trouvée'})

@app.route('/download_results/<task_id>')
def download_results(task_id):
    """Télécharger les résultats"""
    # Implementation pour télécharger les fichiers de résultats
    return jsonify({'success': False, 'message': 'Fonctionnalité en cours de développement'})

def add_log(task_id, message):
    """Ajouter un message au log d'une tâche"""
    timestamp = datetime.now().strftime('%H:%M:%S')
    log_entry = f"[{timestamp}] {message}"
    
    if task_id not in logs_storage:
        logs_storage[task_id] = []
    
    logs_storage[task_id].append(log_entry)

def update_progress(task_id, progress):
    """Mettre à jour le progrès d'une tâche"""
    if task_id in processing_tasks:
        processing_tasks[task_id]['progress'] = progress

def process_mdm_task(task_id, config_data):
    """Traiter une tâche MDM personnalisée"""
    try:
        add_log(task_id, '🚀 Démarrage du traitement MDM personnalisé...')
        update_progress(task_id, 10)
        
        # Simulation du traitement
        time.sleep(1)
        add_log(task_id, '⚙️ Configuration créée')
        update_progress(task_id, 30)
        
        time.sleep(1)
        add_log(task_id, '🔧 Moteur MDM initialisé')
        update_progress(task_id, 50)
        
        time.sleep(2)
        add_log(task_id, '📊 Traitement des données en cours...')
        update_progress(task_id, 80)
        
        time.sleep(1)
        add_log(task_id, '✅ Traitement terminé!')
        
        # Résultats simulés
        result = {
            'input_records': 150,
            'golden_records': 89,
            'execution_time': 5.2,
            'output_files': ['resultat_1.csv', 'resultat_2.csv'],
            'memory_usage': '128MB'
        }
        
        processing_tasks[task_id]['result'] = result
        processing_tasks[task_id]['status'] = 'termine'
        processing_tasks[task_id]['end_time'] = datetime.now()
        update_progress(task_id, 100)
        
        add_log(task_id, f'📊 Enregistrements d\'entrée: {result["input_records"]}')
        add_log(task_id, f'🏆 Enregistrements dorés: {result["golden_records"]}')
        add_log(task_id, f'⏱️ Temps d\'exécution: {result["execution_time"]:.2f} secondes')
        
    except Exception as e:
        add_log(task_id, f'❌ Erreur: {str(e)}')
        processing_tasks[task_id]['status'] = 'erreur'
        processing_tasks[task_id]['error'] = str(e)

def process_example(task_id, example_id):
    """Traiter un exemple prédéfini"""
    try:
        examples = {
            1: process_example_1,
            2: process_example_2,
            3: process_example_3,
            4: process_example_4,
            5: process_example_5
        }
        
        if example_id in examples:
            examples[example_id](task_id)
        else:
            add_log(task_id, f'❌ Exemple {example_id} non trouvé')
            processing_tasks[task_id]['status'] = 'erreur'
            
    except Exception as e:
        add_log(task_id, f'❌ Erreur dans l\'exemple {example_id}: {str(e)}')
        processing_tasks[task_id]['status'] = 'erreur'

def process_example_1(task_id):
    """Exemple 1: Traitement CSV basique"""
    add_log(task_id, '=' * 60)
    add_log(task_id, 'EXEMPLE 1: Traitement CSV basique')
    add_log(task_id, '=' * 60)
    
    add_log(task_id, '📁 Création des données d\'échantillon...')
    update_progress(task_id, 20)
    time.sleep(1)
    
    add_log(task_id, '⚙️ Configuration du moteur MDM...')
    update_progress(task_id, 40)
    time.sleep(1)
    
    add_log(task_id, '🔧 Initialisation du traitement...')
    update_progress(task_id, 60)
    time.sleep(1)
    
    add_log(task_id, '📊 Traitement des enregistrements...')
    update_progress(task_id, 80)
    time.sleep(2)
    
    add_log(task_id, '✅ Traitement terminé!')
    result = {
        'input_records': 5,
        'golden_records': 3,
        'execution_time': 2.1,
        'example': 'CSV basique'
    }
    
    processing_tasks[task_id]['result'] = result
    processing_tasks[task_id]['status'] = 'termine'
    processing_tasks[task_id]['end_time'] = datetime.now()
    update_progress(task_id, 100)
    
    add_log(task_id, '📊 Enregistrements dorés:')
    add_log(task_id, '   • Jean Dupont - Priorité source')
    add_log(task_id, '   • Marie Martin - Plus récent')
    add_log(task_id, '   • Robert Johnson - Score qualité')

def process_example_2(task_id):
    """Exemple 2: Configuration de similarité avancée"""
    add_log(task_id, '=' * 60)
    add_log(task_id, 'EXEMPLE 2: Configuration de similarité avancée')
    add_log(task_id, '=' * 60)
    
    add_log(task_id, '🔍 Test des méthodes de similarité...')
    update_progress(task_id, 25)
    time.sleep(1)
    
    test_pairs = [
        ("Jean Dupont", "Jon Dupont"),
        ("123 Rue Principale", "123 Rue Princ."),
        ("jean@email.com", "jean@gmail.com")
    ]
    
    for str1, str2 in test_pairs:
        add_log(task_id, f'Comparaison: \'{str1}\' vs \'{str2}\'')
        add_log(task_id, '  jarowinkler    : 0.892')
        add_log(task_id, '  levenshtein    : 0.756')
        add_log(task_id, '  exact          : 0.000')
        update_progress(task_id, min(90, processing_tasks[task_id]['progress'] + 20))
        time.sleep(0.5)
    
    add_log(task_id, '💡 Recommandations de méthodes:')
    add_log(task_id, 'name           : jarowinkler, exact')
    add_log(task_id, 'address        : levenshtein, cosine')
    add_log(task_id, 'phone          : levenshtein')
    
    processing_tasks[task_id]['status'] = 'termine'
    processing_tasks[task_id]['end_time'] = datetime.now()
    update_progress(task_id, 100)

def process_example_3(task_id):
    """Exemple 3: Configuration PostgreSQL"""
    add_log(task_id, '=' * 60)
    add_log(task_id, 'EXEMPLE 3: Configuration PostgreSQL avancée')
    add_log(task_id, '=' * 60)
    
    add_log(task_id, '⚙️ Création de la configuration PostgreSQL...')
    update_progress(task_id, 30)
    time.sleep(1)
    
    add_log(task_id, '✅ Configuration PostgreSQL créée: config_postgresql.yaml')
    update_progress(task_id, 60)
    
    add_log(task_id, '🔍 Validation de la configuration:')
    add_log(task_id, '   Type de source: postgresql')
    add_log(task_id, '   Méthode de blocage: fuzzy')
    add_log(task_id, '   Configurations de similarité: 4')
    add_log(task_id, '   Règles de survivance: 4')
    update_progress(task_id, 90)
    
    add_log(task_id, '📊 Connecteurs de base de données disponibles:')
    add_log(task_id, '   • PostgreSQL, MySQL, SQLite, CSV, Excel')
    
    processing_tasks[task_id]['status'] = 'termine'
    processing_tasks[task_id]['end_time'] = datetime.now()
    update_progress(task_id, 100)

def process_example_4(task_id):
    """Exemple 4: Optimisation des performances"""
    add_log(task_id, '=' * 60)
    add_log(task_id, 'EXEMPLE 4: Optimisation des performances')
    add_log(task_id, '=' * 60)
    
    add_log(task_id, '📊 Génération d\'un jeu de données volumineux...')
    update_progress(task_id, 20)
    time.sleep(2)
    
    add_log(task_id, '📊 Jeu de données généré: 10,000 enregistrements')
    update_progress(task_id, 40)
    
    add_log(task_id, '⚙️ Configuration optimisée pour les performances:')
    add_log(task_id, '   • Méthode de blocage: exact (plus rapide)')
    add_log(task_id, '   • Taille de lot: 5000')
    add_log(task_id, '   • Parallélisation: 4 processus')
    update_progress(task_id, 60)
    
    add_log(task_id, '🚀 Démarrage du traitement optimisé...')
    time.sleep(3)  # Simulation du traitement
    update_progress(task_id, 90)
    
    add_log(task_id, '⚡ Résultats de performance:')
    add_log(task_id, '   Enregistrements d\'entrée: 10,000')
    add_log(task_id, '   Enregistrements dorés: 8,520')
    add_log(task_id, '   Temps de traitement: 15.70 secondes')
    add_log(task_id, '   Enregistrements/seconde: 637.0')
    add_log(task_id, '   Utilisation mémoire: 512MB')
    
    processing_tasks[task_id]['status'] = 'termine'
    processing_tasks[task_id]['end_time'] = datetime.now()
    update_progress(task_id, 100)

def process_example_5(task_id):
    """Exemple 5: Stratégies de survivance personnalisées"""
    add_log(task_id, '=' * 60)
    add_log(task_id, 'EXEMPLE 5: Stratégies de survivance personnalisées')
    add_log(task_id, '=' * 60)
    
    add_log(task_id, '📊 Données d\'exemple pour les tests de survivance:')
    add_log(task_id, '   5 enregistrements avec indicateurs de qualité')
    update_progress(task_id, 30)
    time.sleep(1)
    
    add_log(task_id, '⚙️ Configuration de survivance avancée:')
    add_log(task_id, '   • Plus récent: last_updated')
    add_log(task_id, '   • Priorité source: CRM > ERP > Manuel > Import')
    add_log(task_id, '   • Plus long: address')
    add_log(task_id, '   • Valeur plus élevée: quality_score')
    update_progress(task_id, 60)
    
    add_log(task_id, '🔍 Test de survivance sur cluster Jean Dupont:')
    add_log(task_id, '   2 enregistrements dans le cluster')
    update_progress(task_id, 80)
    time.sleep(1)
    
    add_log(task_id, '✅ Résultat de survivance:')
    add_log(task_id, '   ID du survivant: 1')
    add_log(task_id, '   Logique de résolution: Vérification + Score qualité')
    add_log(task_id, '   Score de confiance: 0.92')
    
    add_log(task_id, '📋 Champs de l\'enregistrement doré:')
    add_log(task_id, '   first_name: Jean')
    add_log(task_id, '   last_name: Dupont')
    add_log(task_id, '   data_source: CRM')
    
    add_log(task_id, '🏆 Résumé final:')
    add_log(task_id, '   Enregistrements d\'entrée: 5')
    add_log(task_id, '   Enregistrements dorés: 3')
    
    processing_tasks[task_id]['status'] = 'termine'
    processing_tasks[task_id]['end_time'] = datetime.now()
    update_progress(task_id, 100)

if __name__ == '__main__':
    print("🚀 Démarrage de l'application EasyMDM Advanced")
    print("📱 Interface web disponible sur: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)