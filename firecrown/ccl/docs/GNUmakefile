PUMLIMAGE=miy4/plantuml:latest
PUML=lima nerdctl run -v=$(CURDIR):/work -w /work --rm --interactive $(PUMLIMAGE)

all: updateable.png twopoint.png statistic.png

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
