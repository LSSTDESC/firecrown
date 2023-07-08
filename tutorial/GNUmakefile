.SUFFIXES: .html .qmd
.PHONY: clean

.qmd.html:
	quarto render $<

all: introduction_to_firecrown.html intro_article.html
introduction_to_firecrown.html: custom.scss clean_title_slide.html introduction_to_firecrown.qmd
intro_article.html: custom.scss intro_article.qmd

clean:
	rm -rf introduction_to_firecrown.html introduction_to_firecrown_files/
	rm -rf intro_article.{html,pdf,epub} intro_article_files/
