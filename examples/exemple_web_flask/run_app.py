#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de lancement pour l'application EasyMDM Advanced
Ce script configure l'environnement et lance l'application Flask
"""

import os
import sys
import subprocess
import webbrowser
import time
from pathlib import Path

def check_requirements():
    """Vérifier et installer les dépendances"""
    requirements_file = Path(__file__).parent / 'requirements_flask.txt'
    
    if not requirements_file.exists():
        print("❌ Fichier requirements_flask.txt non trouvé")
        return False
    
    try:
        # Vérifier si Flask est installé
        import flask
        print("✅ Flask déjà installé")
    except ImportError:
        print("📦 Installation des dépendances...")
        try:
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install', 
                '-r', str(requirements_file)
            ])
            print("✅ Dépendances installées avec succès")
        except subprocess.CalledProcessError as e:
            print(f"❌ Erreur lors de l'installation: {e}")
            print("💡 Essayez d'installer manuellement avec:")
            print(f"   pip install -r {requirements_file}")
            return False
    
    return True

def setup_directories():
    """Créer les dossiers nécessaires"""
    directories = ['uploads', 'outputs', 'configs', 'templates', 'static/css']
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("✅ Dossiers créés")

def launch_browser(url, delay=2):
    """Lancer le navigateur avec un délai"""
    def open_browser():
        time.sleep(delay)
        try:
            webbrowser.open(url)
        except Exception as e:
            print(f"⚠️  Impossible d'ouvrir le navigateur automatiquement: {e}")
            print(f"🌐 Ouvrez manuellement l'URL: {url}")
    
    import threading
    threading.Thread(target=open_browser, daemon=True).start()

def main():
    """Fonction principale"""
    print("🚀 EasyMDM Advanced - Démarrage de l'application")
    print("=" * 60)
    
    # Vérifier les dépendances
    print("🔍 Vérification des dépendances...")
    if not check_requirements():
        sys.exit(1)
    
    # Créer les dossiers
    print("📁 Configuration des dossiers...")
    setup_directories()
    
    # Informations de démarrage
    host = '127.0.0.1'
    port = 5000
    url = f'http://{host}:{port}'
    
    print(f"\n✅ Configuration terminée!")
    print(f"🌐 L'application sera disponible sur: {url}")
    print(f"📱 Interface moderne avec toutes les fonctionnalités EasyMDM")
    print("\n💡 Fonctionnalités disponibles:")
    print("   • Téléchargement de fichiers CSV par glisser-déposer")
    print("   • Configuration interactive des paramètres MDM")
    print("   • 5 exemples prédéfinis avec documentation")
    print("   • Suivi en temps réel du traitement")
    print("   • Interface responsive et moderne")
    print("   • Journal d'exécution détaillé")
    
    print(f"\n🔄 Démarrage du serveur...")
    print("⏹️  Pour arrêter l'application, utilisez Ctrl+C")
    print("=" * 60)
    
    # Lancer le navigateur automatiquement
    launch_browser(url)
    
    try:
        # Importer et lancer l'application
        from app import app
        app.run(
            host=host,
            port=port,
            debug=False,  # Désactiver le debug en production
            use_reloader=False  # Éviter le double démarrage
        )
    except ImportError as e:
        print(f"❌ Erreur lors de l'importation de l'application: {e}")
        print("💡 Assurez-vous que le fichier app.py est présent")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⏹️  Application arrêtée par l'utilisateur")
        print("👋 Merci d'avoir utilisé EasyMDM Advanced!")
    except Exception as e:
        print(f"\n❌ Erreur lors du démarrage: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()