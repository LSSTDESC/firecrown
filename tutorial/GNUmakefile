.SUFFIXES: .html .qmd
.PHONY: clean

.qmd.html:
	quarto render $<

all: introduction_to_firecrown.html intro_article.html development_example.qmd
introduction_to_firecrown.html: custom.scss clean_title_slide.html introduction_to_firecrown.qmd \
	_what_is_firecrown.qmd \
	_basic_firecrown_concepts.qmd \
	_high_level_firecrown_classes.qmd \
	_building_blocks_for_the_gaussfamily_likelihoods.qmd \
	_development_workflow.qmd \
	_examples_in_the_repository.qmd \
	_installation_modes.qmd \
	 _developer_installation.qmd \
	 _setting_up_a_shell_session_for_development.qmd \
	 _building_firecrown.qmd \
	 _code_development_hygiene.qmd

intro_article.html: custom.scss intro_article.qmd

clean:
	rm -rf introduction_to_firecrown.html introduction_to_firecrown_files/
	rm -rf intro_article.{html,pdf,epub} intro_article_files/
