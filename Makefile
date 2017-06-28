PANDOC = "D:\Program Files (x86)\Pandoc\pandoc.exe"
PDFVIEW = "D:\texlive\2015\tlpkg\tlpsv\psv.bat"

LATEXFLAGS = -H header.tex -B begin.tex -A append.tex -s

.PHONY: clear
clear:
	-rm -rf *.aux *.bbl *.blg *.docx *.dvi *.log *.pdf *.out E*.tex

E01.pdf: E01.md
	$(PANDOC) $(LATEXFLAGS) E01.md -o E01.pdf
E02.pdf: E02.md
	$(PANDOC) $(LATEXFLAGS) E02.md -o E02.pdf
E03.pdf: E03.md
	$(PANDOC) $(LATEXFLAGS) E03.md -o E03.pdf
E04.pdf: E04.md
	$(PANDOC) $(LATEXFLAGS) E04.md -o E04.pdf
E05.pdf: E05.md
	$(PANDOC) $(LATEXFLAGS) E05.md -o E05.pdf
E06.pdf: E06.md
	$(PANDOC) $(LATEXFLAGS) E06.md -o E06.pdf
E07.pdf: E07.md
	$(PANDOC) $(LATEXFLAGS) E07.md -o E07.pdf
E08.pdf: E08.md
	$(PANDOC) $(LATEXFLAGS) E08.md -o E08.pdf
E09.pdf: E09.md
	$(PANDOC) $(LATEXFLAGS) E09.md -o E09.pdf
E10.pdf: E10.md
	$(PANDOC) $(LATEXFLAGS) E10.md -o E10.pdf
E11.pdf: E11.md
	$(PANDOC) $(LATEXFLAGS) E11.md -o E11.pdf
E12.pdf: E12.md
	$(PANDOC) $(LATEXFLAGS) E12.md -o E12.pdf
E13.pdf: E13.md
	$(PANDOC) $(LATEXFLAGS) E13.md -o E13.pdf
E14.pdf: E14.md
	$(PANDOC) $(LATEXFLAGS) E14.md -o E14.pdf
E15.pdf: E15.md
	$(PANDOC) $(LATEXFLAGS) E15.md -o E15.pdf
E16.pdf: E16.md
	$(PANDOC) $(LATEXFLAGS) E16.md -o E16.pdf
E17.pdf: E17.md
	$(PANDOC) $(LATEXFLAGS) E17.md -o E17.pdf
E18.pdf: E18.md
	$(PANDOC) $(LATEXFLAGS) E18.md -o E18.pdf
E19.pdf: E19.md
	$(PANDOC) $(LATEXFLAGS) E19.md -o E19.pdf
E20.pdf: E20.md
	$(PANDOC) $(LATEXFLAGS) E20.md -o E20.pdf
