PANDOC = pandoc
LATEXFLAGS = -H header.tex -B begin.tex -A append.tex -s

LATEXDEPS = header.tex begin.tex append.tex
TARGETS = E01 E02 E03 E04 E05 E06 E07 E08 E09 E10 \
          E11 E12 E13 E14 E15 E16 E17 E18 E19 E20
TEX = $(addsuffix .tex, $(TARGETS))
PDF = $(addsuffix .pdf, $(TARGETS))
DOCX = $(addsuffix .docx, $(TARGETS))

.PHONY: list
list:
	@$(MAKE) -pRrq -f $(lastword $(MAKEFILE_LIST)) : 2>/dev/null | awk -v RS= -F: '/^# File/,/^# Finished Make data base/ {if ($$1 !~ "^[#.]") {print $$1}}' | sort | egrep -v -e '^[^[:alnum:]]' -e '^$@$$' | xargs

.PHONY: all
all: $(TEX) $(PDF) $(DOCX)

.PHONY: cleanall
cleanall: clean
	$(RM) *.pdf E*.tex *.docx

.PHONY: clean
clean:
	$(RM) *.aux *.bbl *.blg *.dvi *.log *.out

$(TEX): %.tex: %.md $(LATEXDEPS)
	$(PANDOC) $(LATEXFLAGS) $< -o $@

$(PDF): %.pdf: %.md $(LATEXDEPS)
	$(PANDOC) $(LATEXFLAGS) $< -o $@

$(DOCX): %.docx: %.md
	$(PANDOC) $? -o $@

summary.pdf: summary.md sumbegin.tex sumheader.tex append.tex
	$(PANDOC) -H sumheader.tex -B sumbegin.tex -A sumappend.tex -s summary.md -o summary.pdf --latex-engine=xelatex
