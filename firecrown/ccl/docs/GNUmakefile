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

likelihood.png: skinparams.iuml updateable_frag.iuml
statistic.png: skinparams.iuml statistic_frag.iuml updateable_frag.iuml
twopoint.png: skinparams.iuml statistic_frag.iuml updateable_frag.iuml
updateable.png: skinparams.iuml updateable_frag.iuml
