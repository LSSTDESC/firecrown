.SUFFIXES: .html .qmd
.PHONY: clean

.qmd.html:
	quarto render $<

all: introduction_to_firecrown.html
introduction_to_firecrown.html: custom.scss

clean:
	rm -f *.html
