ifeq (true,$(CONTAINER))
	PUMLIMAGE=miy4/plantuml:latest
	PUML=lima nerdctl run -v=$(CURDIR):/work -w /work --rm --interactive $(PUMLIMAGE)
else
	PUML=plantuml
endif

SOURCES = $(wildcard *.puml)
PNGS = $(patsubst %.puml,%.png,$(SOURCES))
PDFS = $(patsubst %.puml,%.pdf,$(SOURCES))

all: png pdf
png: $(PNGS)
pdf: $(PDFS)

.PHONY : clean
.SUFFIXES : .puml .eps .pdf .png

.puml.png:
	$(PUML) -tpng $<

.puml.eps:
	$(PUML) -teps $<
	
.eps.pdf:
	epstopdf $<

clean:
	rm -f *.eps *.pdf *.png

$(PNGS): skinparams.iuml
$(PDFS): skinparams.iuml

gaussian-family.png: updatable_frag.iuml likelihood_frag.iuml
likelihood.png: updatable_frag.iuml  likelihood_frag.iuml
statistic.png: statistic_frag.iuml updatable_frag.iuml
twopoint.png: statistic_frag.iuml updatable_frag.iuml
updatable.png: updatable_frag.iuml
