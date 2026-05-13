# 🧪 Point Group Project

### Interactive Organic Molecular Symmetry Analysis in Python

Molecular symmetry is one of those chemistry topics that is both fascinating and notoriously difficult to visualize. On paper, concepts like rotational axes, mirror planes, inversion centers, and character tables can quickly become abstract — especially when trying to imagine them in three dimensions 🤯📐 

During our second year at EPFL, we wanted to change that by creating a tool that makes molecular symmetry feel interactive, intuitive, and genuinely fun to explore.

Point Group Project is an educational computational chemistry application that automatically analyzes the symmetry of a molecule starting from something as simple as its IUPAC name 🧬✨ Instead of manually searching for symmetry elements, users can instantly generate a fully optimized 3D molecular structure, detect its symmetry operations, determine its point group, and visualize everything directly in an interactive 3D environment 🌍🔬

What excited us most about this project is the way it combines several fields into one experience: chemistry ⚛️, mathematics 📊, programming 💻, scientific visualization 🎨, and molecular modeling 🧪. Behind the scenes, the program uses real computational chemistry workflows: molecules are retrieved from chemical databases, optimized in 3D, analyzed through symmetry operations, and finally rendered interactively with custom graphical overlays for axes, planes, and inversion centers.

---
✨ Features
🧬 Automatic 3D molecule generation from an IUPAC name
🔄 Detection of molecular symmetry operations
📐 Automatic point group determination
🎨 Interactive visualization of symmetry axes and mirror planes
🌐 Real-time 3D molecular rendering
📚 Character table visualization for selected point groups
⚛️ Analysis of molecular properties:
- chirality
- polarity
- IR activity
- Raman activity

---

## 🚀 Technologies

- **RDKit** — molecular generation & optimization
- **pymatgen** — symmetry analysis
- **PubChemPy** — molecular database access
- **Streamlit** — interactive interface
- **py3Dmol** — 3D visualization

---

## ▶️ Launch the Project

Clone the repository:

```bash
git clone https://github.com/shuyijuliettejia/point-group_project.git
cd point-group_project
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the visualization interface:

```bash
streamlit run scripts/visualisation.py
```

Optional - If you need jupyter lab, install it:

```bash
pip install jupyterlab
```

Optional - Using Conda Environments
If you prefer using Conda for Python environments, you can create an empty environment before installing as so:

```bash
conda create -n #your_env_name python=3.10
conda activate #your_env_name
```
---

## 🎯 Why This Project Matters

This project aims to make molecular symmetry easier to visualize and far more interactive than the traditional textbook approach 📚✨ Instead of memorizing abstract symmetry operations, students can directly explore them in 3D and develop a more intuitive understanding of how molecules behave in space 🧬🌍 Ultimately, our goal is to turn a concept that can sometimes feel intimidating into something engaging, accessible, and even fun to learn 🚀⚛️

---

## 👩‍🔬 Authors

Developed at École Polytechnique Fédérale de Lausanne by:
- Julie Schweizer
- Margaux Bourhis
- Shuyi Jia

---

## 📚 Educational Project

This repository was created as part of a chemistry programming project focused on molecular symmetry and point group analysis.
