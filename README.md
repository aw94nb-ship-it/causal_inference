# Causal Inference Notes

A Jupyter Book documenting my learning journey through causal inference, featuring theory, code examples, and practical applications.

## 📚 View the Book

Visit the live website: **https://anniewang.github.io/causal_inference/** (once deployed)

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip

### Installation

1. Clone this repository:
```bash
git clone https://github.com/anniewang/causal_inference.git
cd causal_inference
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Building the Book Locally

Build the HTML version:
```bash
jupyter-book build .
```

The built website will be in `_build/html/`. Open `_build/html/index.html` in your browser.

### Cleaning Build Files

To remove previous builds:
```bash
jupyter-book clean .
```

For a complete clean (including cached outputs):
```bash
jupyter-book clean --all .
```

## 📝 Adding New Content

### Structure

The book is organized as follows:
```
causal_inference/
├── intro.md                    # Homepage
├── _config.yml                 # Book configuration
├── _toc.yml                   # Table of contents
├── chapters/
│   ├── fundamentals/          # Basic concepts
│   ├── methods/              # Causal inference methods
│   ├── advanced/             # Advanced topics
│   └── resources.md          # References and resources
└── references.bib            # Bibliography
```

### Adding a New Page

1. Create a new Markdown file (`.md`) or Jupyter notebook (`.ipynb`) in the appropriate chapter folder
2. Add it to `_toc.yml` in the appropriate section
3. Rebuild the book

Example entry in `_toc.yml`:
```yaml
chapters:
  - file: chapters/methods
    sections:
      - file: chapters/methods/your-new-page
```

### Writing Content

#### Markdown Files
Use standard Markdown with MyST extensions for:
- **Math**: `$inline math$` or `$$display math$$`
- **Citations**: `{cite}`pearl2009causality``
- **Cross-references**: `{ref}`label``
- **Admonitions**:
  ```markdown
  :::{note}
  This is a note
  :::
  ```

#### Jupyter Notebooks
- Include explanatory text in Markdown cells
- Add code examples in code cells
- The notebooks will execute automatically when building (configurable in `_config.yml`)

## 🔄 Automatic Deployment

The website automatically deploys to GitHub Pages when you push to the `main` branch.

### First-Time Setup

1. Go to your repository Settings
2. Navigate to Pages (under "Code and automation")
3. Under "Build and deployment":
   - Source: Select "GitHub Actions"
4. Push to main branch to trigger the first deployment

The site will be available at: `https://yourusername.github.io/causal_inference/`

## 📖 What's Inside

### Fundamentals
- Causality basics
- Potential outcomes framework
- Directed Acyclic Graphs (DAGs)

### Methods
- Randomized Controlled Trials
- Matching methods
- Regression Discontinuity Design
- Instrumental Variables
- Difference-in-Differences

### Advanced Topics
- Sensitivity analysis
- Mediation analysis
- Causal machine learning

### Resources
- Textbooks and papers
- Online courses
- Software tools
- Practice datasets

## 🛠️ Customization

### Update Book Metadata

Edit `_config.yml` to change:
- Book title and author
- Repository links
- Theme options
- Execution settings

### Update Navigation

Edit `_toc.yml` to:
- Reorder chapters
- Add/remove pages
- Create nested sections

## 📦 Dependencies

Main packages:
- `jupyter-book`: Core book building tool
- `matplotlib`: Plotting
- `numpy`, `pandas`: Data manipulation
- `scipy`: Statistical functions

Add additional packages to `requirements.txt` as needed for your analyses.

## 🤝 Contributing

This is a personal learning repository, but feel free to:
- Open issues for errors or suggestions
- Fork and adapt for your own notes
- Share resources in discussions

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

Resources and inspiration from:
- Angrist & Pischke's "Mostly Harmless Econometrics"
- Hernán & Robins' "Causal Inference: What If"
- Pearl's "Causality"
- The broader causal inference community

---

*Happy learning!* 🎓
