import numpy as np
import matplotlib.pyplot as plt

# Chemin vers tes données
PATH = '/home/barrage/SPECT_COSMOS_US/us/COSMOS_EmLines_D.npz'

def inspect_data():
    print(f"--- INSPECTION : {PATH} ---")
    
    try:
        data = np.load(PATH)
        info = data['info']
        
        # 1. Analyse des Colonnes (Pour résoudre le mystère "uz")
        if hasattr(info.dtype, 'names'):
            names = [n.lower() for n in info.dtype.names]
            print(f"Colonnes disponibles : {names}")
            
            # Vérification des bandes
            if 'uz' in names:
                print(">>> ATTENTION : Colonne 'uz' détectée ! (Hypothèse Strömgren confirmée ?)")
            elif 'i' in names and 'z' in names:
                print(">>> Standard : Bandes 'i' et 'z' détectées.")
            else:
                print(">>> ATTENTION : Bandes non standard.")
        
        # 2. Vérification des Flags (CRUCIAL)
        if 'flag' in data.files:
            flags = data['flag']
            print(f"Shape des flags : {flags.shape}")
            
            # On vérifie les 6 premières bandes (0 à 5)
            # flag = 0 signifie "clean"
            flags_subset = flags[:, :6] 
            sum_flags = np.sum(flags_subset, axis=1)
            
            n_total = len(info)
            n_clean = np.sum(sum_flags == 0)
            percent = (n_clean / n_total) * 100
            
            print(f"\nANALYSE QUALITÉ (FLAGS):")
            print(f"Total objets : {n_total}")
            print(f"Objets cleans (sum flags=0) : {n_clean} ({percent:.2f}%)")
            
            if n_clean == 0:
                print("!!! ALERTE ROUGE : Aucun objet n'a flag=0. Vérifiez les indices des canaux.")
        else:
            print("\n!!! ALERTE : Pas de clé 'flag' dans le fichier .npz")

        # 3. Vérification des Magnitudes (pour la coupure à 25)
        if 'i' in names:
            mags = info['i']
            n_mag_cut = np.sum(mags <= 25.0)
            print(f"\nANALYSE MAGNITUDE i:")
            print(f"Objets avec i <= 25.0 : {n_mag_cut}")
            print(f"Min: {np.min(mags):.2f}, Max: {np.max(mags):.2f}")

    except Exception as e:
        print(f"\nERREUR CRITIQUE : {e}")

if __name__ == "__main__":
    inspect_data()