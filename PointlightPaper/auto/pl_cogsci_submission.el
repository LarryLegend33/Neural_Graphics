(TeX-add-style-hook
 "pl_cogsci_submission"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("article" "10pt" "letterpaper")))
   (TeX-run-style-hooks
    "latex2e"
    "article"
    "art10"
    "cogsci"
    "pslatex"
    "apacite"
    "float")
   (LaTeX-add-labels
    "common_motion_phen"
    "sample-table"
    "sample-figure")
   (LaTeX-add-bibliographies
    "pl_cogsci.bib"))
 :latex)

