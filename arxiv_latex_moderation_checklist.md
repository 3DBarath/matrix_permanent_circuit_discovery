# arXiv LaTeX & Moderation Checklist

Tick these off before uploading your `.tar.gz` or `.zip` bundle to avoid automated TeX failures or manual moderation holds.

## LaTeX & Source Files
- [ ] **Upload LaTeX source, not a standalone PDF**
  - arXiv rejects PDF-only uploads if it detects the PDF was generated from LaTeX.
- [ ] **Include the compiled `.bbl` file**
  - arXiv does not run BibTeX; your `.bbl` filename must match your main `.tex` file.
- [ ] **Add `\pdfoutput=1` in the first 5 lines**
  - Forces arXiv to compile with `pdflatex` instead of legacy `latex`.
- [ ] **Fix image paths and case sensitivity**
  - Use relative paths only, avoid spaces in filenames, and match exact casing.
- [ ] **Strip build artifacts and private comments**
  - Delete `.aux`, `.log`, `.out`, `.synctex.gz` files and any draft `TODO` comments (arXiv source is public).
- [ ] **Keep total bundle size under 50 MB**
  - Compress oversized PNG/PDF figures if needed.

## Metadata & Formatting
- [ ] **Match PDF and form metadata exactly**
  - Title, author list, and abstract should match word-for-word.
- [ ] **Clean up the metadata abstract**
  - Remove custom LaTeX macros and `\cite{}` commands; stick to plain text and basic `$...$` math.

## Moderation & Policy
- [ ] **Verify your primary category**
  - Check recent papers in your target category (e.g., `cs.LG`, `cs.CL`) to confirm scope fit.
- [ ] **Check for text overlap and policy compliance**
  - Avoid heavy reuse from prior drafts without citation, and ensure CS surveys/position papers have prior peer review.