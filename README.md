# iop-projekat
Projekat iz predmeta Inteligentna obrada podataka

## Neke ideje

- Elementarni objekti: skalari, vektori, matrice
- Promenljive: Wrappuju elementarne objekte, mogu da budu ulazne, izlazne, mogu da se prave prosto operacijama i funkcijama ciji su operandi (argumenti) druge promenljive.
- Izracunavanje: Jedan skup ulaznih i izlaznih promenljivih. Da bi se izracunavanje pokrenulo moraju da se vezu sve slobodne ulazne promenljive tj. da im se zada vrednost
- Lanci: Niz identicnih iteracija gde je izlaz k-te iteracije ustvari ulaz (k+1)-ve. Lanac je po svojoj prirodi iteracija. Zadaje se koje promenljive se vezuju za koje.
- Kontekst: CPU ili GPU
- Neki skup funkcija koje mogu da se primene nad elementarnim objektima: exp, log, sin, cos, sqrt, abs. (Zadrzati se na nekom manjem skupu jer ce da nam trebaju posebni kerneli za GPU za svaku, mada power copy-paste-a vrv ce da ucini svoje)

Primer:

Napravim nekakav kontekst c koji je vezan za GPU, npr. `auto c = iop::context("GPU");`
Napravim matricu i vektor (promenljive): `auto A = c.matrix(100, 100); auto b = c.vector(100)`;
Napravim promenljivu koju vezujem za recimo sinus matricnog proizvoda: `auto v = iop::sin(A.dot(b));`
Evaluiram jednom: `iop::run({{b, podaci}}, {v});`. Mozda da parametri budu lista parova (promenljiva, poc. vrednost) i izlazna promenljiva a da ovo cudo vrati izracunatu vrednost izlazne promenljive.
