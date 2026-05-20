

import numpy as np
import streamlit as st
import py3Dmol
from stmol import showmol
from molecule_reel import construire_molecule_data


def construire_xyz(molecule_data):
    atomes = molecule_data["atomes"]
    lignes = [str(len(atomes)), molecule_data.get("nom", "molecule")]
    for a in atomes:
        lignes.append(f"{a['element']}  {a['x']:.4f}  {a['y']:.4f}  {a['z']:.4f}")
    return "\n".join(lignes)


def ajouter_axes(vue, molecule_data):
    for axe in molecule_data.get("axes", []):
        d = np.array(axe["direction"], dtype=float)
        d = d / np.linalg.norm(d)
        longueur = 2.0

        debut = (-d * longueur).tolist()
        fin   = ( d * longueur).tolist()

        
        vue.addCylinder({
            "start":  {"x": debut[0], "y": debut[1], "z": debut[2]},
            "end":    {"x": fin[0],   "y": fin[1],   "z": fin[2]},
            "radius": 0.05,
            "color":  "#E24B4A",
            "opacity": 0.9,
        })

       
        vue.addSphere({
            "center": {"x": fin[0], "y": fin[1], "z": fin[2]},
            "radius": 0.12,
            "color":  "#E24B4A",
        })

        
        vue.addLabel(axe["label"], {
            "position":        {"x": fin[0]*1.2, "y": fin[1]*1.2, "z": fin[2]*1.2},
            "fontSize":        14,
            "fontColor":       "#E24B4A",
            "backgroundColor": "transparent",
        })

 
COULEURS_PLAN = {"σh": "#3B8BD4", "σv": "#EF9F27", "σd": "#1D9E75"}

def ajouter_plans(vue, molecule_data):
    """
    Dessine chaque plan σ comme deux triangles semi-transparents.
    On calcule les 4 coins depuis la normale (numpy),
    puis py3Dmol dessine les triangles.
    """
    for plan in molecule_data.get("plans", []):
        n = np.array(plan["normale"], dtype=float)
        n = n / np.linalg.norm(n)
        couleur = COULEURS_PLAN.get(plan["type"], "#888888")

        
        ref = np.array([1, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1, 0])
        u   = ref - np.dot(ref, n) * n
        u   = u / np.linalg.norm(u)
        v   = np.cross(n, u)

        s = 1.6
        A = ( u + v) * s
        B = (-u + v) * s
        C = (-u - v) * s
        D = ( u - v) * s

        def pt(c):
            return {"x": float(c[0]), "y": float(c[1]), "z": float(c[2])}

      
        for triangle in [(A, B, C), (A, C, D)]:
            vue.addCustom({
                "vertexColors": [couleur, couleur, couleur],
                "vertices":     [pt(triangle[0]), pt(triangle[1]), pt(triangle[2])],
                "normal_vector": pt(n),
                "opacity":      0.25,
            })

        
        for depart, arrivee in [(A,B),(B,C),(C,D),(D,A)]:
            vue.addCylinder({
                "start":  pt(depart),
                "end":    pt(arrivee),
                "radius": 0.02,
                "color":  couleur,
                "opacity": 0.6,
            })

        # Étiquette au centre du plan
        centre = (A + C) / 2
        vue.addLabel(plan["label"], {
            "position":        pt(centre),
            "fontSize":        12,
            "fontColor":       couleur,
            "backgroundColor": "transparent",
        })


def ajouter_inversion(vue):
    vue.addSphere({
        "center":  {"x": 0, "y": 0, "z": 0},
        "radius":  0.15,
        "color":   "#D4537E",
        "opacity": 0.9,
    })
    vue.addLabel("i", {
        "position":        {"x": 0.2, "y": 0.2, "z": 0.2},
        "fontSize":        14,
        "fontColor":       "#D4537E",
        "backgroundColor": "transparent",
    })


def deduire_proprietes(molecule_data):
    import re
    pg = molecule_data.get("point_group", "C1")

    groupes_chiraux = {"C1","C2","C3","C4","C5","C6","D2","D3","D4","D5","D6","T","O","I"}
    chiral  = molecule_data.get("chiral",       pg in groupes_chiraux)
    polaire = molecule_data.get("polar",        bool(re.match(r'^C\d+(v)?$', pg)))
    ir      = molecule_data.get("ir_active",    True)
    raman   = molecule_data.get("raman_active", True)

    if molecule_data.get("inversion", False):
        ir_txt    = "Partiel (règle d'exclusion)"
        raman_txt = "Partiel (règle d'exclusion)"
    else:
        ir_txt    = "Yes" if ir    else "No"
        raman_txt = "Yes" if raman else "No"

    return chiral, polaire, ir_txt, raman_txt


def lancer_interface():
    st.set_page_config(page_title="Symétrie moléculaire", layout="wide")
    st.title("Molecular symmetry visualisator")

    iupac_name = st.text_input("Enter a IUPAC name of organic compound", placeholder="ex: water, ammonia...")

    if "molecule_data" not in st.session_state:
        st.session_state.molecule_data = None

    if st.button("Analyse"):
        with st.spinner("Searching..."):
            try:
                st.session_state.molecule_data = construire_molecule_data(iupac_name)
            except ValueError as e:
                st.error(str(e))
                st.stop()
    molecule_data = st.session_state.molecule_data
    
    if molecule_data is not None:
        st.markdown(f"**{molecule_data['nom']}** — `{molecule_data.get('formule', '')}` "f"— point group : **{molecule_data['point_group']}**")

        col_vue, col_info = st.columns([3, 1])

        with col_info:
            st.markdown("### Display")
            show_axes   = st.checkbox("Axes Cn",    value=True)
            show_plans  = st.checkbox("Plans σ",    value=True)
            show_labels = st.checkbox("Atom's name", value=True)
            style_mol   = st.selectbox("Molecule style", ["stick", "sphere", "line"])

            st.markdown("---")
            st.markdown("### Legend of plans")
            st.markdown("🟠 **σv** — vertical  \n🔵 **σh** — horizontal  \n🟢 **σd** — diedral  \n🔴 **axe Cn**")

            st.markdown("---")
            st.markdown("### Properties")
            chiral, polaire, ir_txt, raman_txt = deduire_proprietes(molecule_data)
            st.metric("Chirality",   "Chiral"  if chiral  else "Achiral")
            st.metric("Polarity",    "Polar"  if polaire else "Apolar")
            st.metric("IR actif",    ir_txt)
            st.metric("Raman actif", raman_txt)
            st.markdown("---")
            st.markdown("### Character table")
            from irreps import get_character_table
            table = get_character_table(molecule_data["point_group"])
            if table:
                st.markdown(f"**Class** : {' | '.join(table['classes'])}")
                for irrep, chars in table["irreps"].items():
                    st.text(f"{irrep:<6} {chars}")
            else:
                st.warning("Table not available for this point group")

            if molecule_data.get("inversion"):
                st.info("Centre d'inversion (i) — point rose au centre")

        with col_vue:
            
            vue = py3Dmol.view(width=700, height=500)

            
            vue.addModel(construire_xyz(molecule_data), "xyz")
            vue.setStyle({}, {
        "stick": {"radius": 0.12, "colorscheme": "Jmol"},
        "sphere": {"scale": 0.3, "colorscheme": "Jmol"}
    })

            
            if show_axes:
                ajouter_axes(vue, molecule_data)
            if show_plans:
                ajouter_plans(vue, molecule_data)
            if molecule_data.get("inversion"):
                ajouter_inversion(vue)

          
            if show_labels:
                for a in molecule_data["atomes"]:
                    vue.addLabel(a["element"], {
                        "position":          {"x": a["x"], "y": a["y"], "z": a["z"]},
                        "fontSize":          12,
                        "fontColor":         "white",
                        "backgroundColor":   "#333333",
                        "backgroundOpacity": 0.6,
                    })

            vue.zoomTo()
          
            showmol(vue, height=500, width=700)

    st.caption("Clic-glisser pour tourner · Molette pour zoomer · Double-clic pour centrer")

if __name__ == "__main__":
    lancer_interface()
