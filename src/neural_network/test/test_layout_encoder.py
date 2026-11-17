import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn

# L'import du layout_encoder peut échouer à cause de bugs dans les dépendances
# (ex: attention.py hérite de torch.nn au lieu de torch.nn.Module).
# On capture l'erreur pour que le test puisse s'exécuter et rapporter les problèmes.
try:
    from src.neural_network.layout_analysis.layout_encoder import Aifi, Ccff, Efficient_hybrid_encoder
    IMPORT_OK = True
    IMPORT_ERROR = None
except Exception as e:
    IMPORT_OK = False
    IMPORT_ERROR = e
    Aifi = Ccff = Efficient_hybrid_encoder = None


def import_test():
    """Vérifie que le module layout_encoder peut être importé sans erreur."""
    if IMPORT_OK:
        print("Import du layout_encoder : OK")
        return True
    else:
        print(f"ÉCHEC import layout_encoder : {type(IMPORT_ERROR).__name__} : {IMPORT_ERROR}")
        return False


def aifi_instantiation_test():
    """Vérifie que la classe Aifi peut être instanciée sans erreur."""
    if not IMPORT_OK:
        print("SKIP (import échoué)")
        return False
    try:
        aifi = Aifi()
        print("Aifi instanciée avec succès")
        return True
    except Exception as e:
        print(f"ÉCHEC instanciation Aifi : {type(e).__name__} : {e}")
        return False


def aifi_forward_test():
    """Teste le forward de Aifi avec une entrée 4D (batch, channels, H, W)."""
    if not IMPORT_OK:
        print("SKIP (import échoué)")
        return False
    try:
        aifi = Aifi()
        # entrée : batch=2, channels=512, hauteur=8, largeur=8
        x = torch.randn(2, 512, 8, 8)
        out = aifi.forward(x)
        print(f"Aifi forward OK, shape de sortie : {tuple(out.shape)}")
        return True
    except Exception as e:
        print(f"ÉCHEC forward Aifi : {type(e).__name__} : {e}")
        return False


def ccff_instantiation_test():
    """Vérifie que la classe Ccff peut être instanciée."""
    if not IMPORT_OK:
        print("SKIP (import échoué)")
        return False
    try:
        ccff = Ccff()
        print("Ccff instanciée avec succès")
        return True
    except Exception as e:
        print(f"ÉCHEC instanciation Ccff : {e}")
        return False


def efficient_hybrid_encoder_instantiation_test():
    """Vérifie que Efficient_hybrid_encoder peut être instanciée avec aifi et ccff."""
    if not IMPORT_OK:
        print("SKIP (import échoué)")
        return False
    try:
        aifi = Aifi()
        ccff = Ccff()
        encoder = Efficient_hybrid_encoder(aifi, ccff)
        assert encoder.aifi is aifi
        assert encoder.ccff is ccff
        print("Efficient_hybrid_encoder instancié avec succès, attributs aifi et ccff présents")
        return True
    except Exception as e:
        print(f"ÉCHEC instanciation Efficient_hybrid_encoder : {type(e).__name__} : {e}")
        return False


def run_all_tests():
    print("=== Tests du layout_encoder ===\n")

    results = {}
    results["import"] = import_test()
    print()
    results["aifi_instantiation"] = aifi_instantiation_test()
    print()
    results["aifi_forward"] = aifi_forward_test()
    print()
    results["ccff_instantiation"] = ccff_instantiation_test()
    print()
    results["efficient_hybrid_encoder_instantiation"] = efficient_hybrid_encoder_instantiation_test()

    print("\n=== Résumé ===")
    for name, ok in results.items():
        status = "OK" if ok else "ÉCHEC"
        print(f"  {name}: {status}")

    all_ok = all(results.values())
    print(f"\nRésultat global : {'TOUS LES TESTS PASSENT' if all_ok else 'DES TESTS ONT ÉCHOUÉ'}")
    return all_ok


if __name__ == "__main__":
    #run_all_tests()
    aifi_instantiation_test()
    aifi_forward_test()
