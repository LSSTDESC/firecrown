
# start_paper
### Jumpstart your DESC paper or note

`start_paper` is intended to make the process of starting to write a DESC paper or note, and later transforming notes into papers, as simple as possible.

## Starting your paper

Download the contents of the `start_paper` repository. We recommend downloading it as a ZIP file rather than cloning the repository; this simplifies the process of versioning your paper in its own repository, if you so desire. You can either do this manually with the "Clone or download" button in GitHub, or automatically by [downloading](https://raw.githubusercontent.com/LSSTDESC/start_paper/master/deploy_from_github_zip.bash) and running [this BASH script](https://github.com/LSSTDESC/start_paper/blob/master/deploy_from_github_zip.bash), as in

```bash
./deploy_from_github_zip.bash MyNewPaper
```

This will download and unzip the `start_paper` files to a new folder called `MyNewPaper/`.

`start_paper` provides templates in various formats: [`Jupyter Notebook`](https://ipython.org/notebook.html) (`main.ipynb`), [`Markdown`](https://github.com/adam-p/Markdown-here/wiki/Markdown-Cheatsheet) (`main.md`), [`reStructuredText`](http://docutils.sourceforge.net/rst.html) (`main.rst`), and latex (`main.tex`). There is also a template [Google Doc](https://docs.google.com/document/d/1ERz_S02Uvc0QkapVx145PrYZT0CRJbkPMmY5T95uMkk/edit?usp=sharing).

## Building your paper

At present, only latex documents require/benefit from the Makefile. `make` or `make help` will display basic options; more detailed documentation on using the Makefile and on writing papers/notes in latex and other formats will be provided in another document (started, appropriately enough, with `start_paper`).


## Updating `start_paper` content

From time to time, there may be updates to the latex support files provided by [`desc-tex`](https://github.com/LSSTDESC/desc-tex), or the `start_paper` templates or Makefile.

```
make templates
```
will download the latest templates and Makefile to a `templates/` directory (i.e., not directly over-writing any files in the main directory).

```
make update
```
will do the same, and will also update `desc-tex` and `mkauthlist`.

## Contributors to `start_paper`

People developing this project:
* Phil Marshall [(@drphilmarshall)](https://github.com/drphilmarshall)
* Alex Drlica-Wagner [(@kadrlica)](https://github.com/kadrlica)
* Adam Mantz [(@abmantz)](https://github.com/abmantz)
* Heather Kelly [(@heather999)](https://github.com/heather999)
* Jonathan Sick [(@jonathansick)](https://github.com/jonathansick)

This is open source software, available under the BSD license. If you are interested in this project, please do drop us a line via the hyperlinked contact names above, or by [writing us an issue](https://github.com/DarkEnergyScienceCollaboration/start_paper/issues).
