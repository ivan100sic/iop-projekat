# iop-projekat
Projekat iz predmeta Inteligentna obrada podataka

## Neke ideje

- Elementarni objekti: skalari, vektori, matrice
- Promenljive: Wrappuju elementarne objekte, mogu da budu ulazne, izlazne, mogu da se prave prosto operacijama i funkcijama ciji su operandi (argumenti) druge promenljive.
- Izracunavanje: Jedan skup ulaznih i izlaznih promenljivih. Da bi se izracunavanje pokrenulo moraju da se vezu sve slobodne ulazne promenljive tj. da im se zada vrednost
- Lanci: Niz identicnih izracunavanja gde je izlaz k-tog izracunavanja ustvari ulaz (k+1)-og. Lanac je po svojoj prirodi izracunavanje. Zadaje se koje promenljive se vezuju za koje.
- Kontekst: CPU, GPU (preko OpenCL) ili mozda CUDA?
- Neki skup funkcija koje mogu da se primene nad elementarnim objektima: exp, log, sin, cos, sqrt, abs. (Zadrzati se na nekom manjem skupu jer ce da nam trebaju posebni kerneli za GPU za svaku, mada power copy-paste-a vrv ce da ucini svoje)

Primer:

- Napravim nekakav kontekst c koji je vezan za GPU, npr. `auto c = iop::context("GPU");`
- Napravim matricu i vektor (promenljive): `auto A = c.matrix(100, 100); auto b = c.vector(100)`;
- Napravim promenljivu koju vezujem za recimo sinus matricnog proizvoda: `auto v = iop::sin(A.dot(b));`
- Evaluiram jednom: `iop::run({{b, podaci}}, {v});`. Mozda da parametri budu lista parova (promenljiva, poc. vrednost) i izlazna promenljiva a da ovo cudo vrati izracunatu vrednost izlazne promenljive.
- Vidim da radi, napravim lanac od 100 iteracija, mozda ovako nekako: `auto ch = iop::chain({A, v}, {A, b});`
- Sad evaluiram lanac. Da bi njega evaluirao moram da mu dam odgovarajucu tuplu ulaza, sa onoliko elemenata i onog tipa kao njegov ulaz/izlaz.
- Formalizovati tip lanca i kako bi trebalo da se radi sa njim. Jako je bitno da ovo bude elegantno.

## Linkovi

- OpenCL specifikacija: https://www.khronos.org/registry/OpenCL/specs/opencl-1.1.pdf
- AMD OpenCL guide: http://developer.amd.com/wordpress/media/2013/12/AMD_OpenCL_Programming_User_Guide2.pdf
- GCN Arhitektura: http://developer.amd.com/wordpress/media/2013/12/AMD_GCN3_Instruction_Set_Architecture_rev1.1.pdf

## Note to self

- Ne moze ovako, ovo je totalno pogresno. Ne postoji apsolutno nikakva
	garancija o konzistentnosti takozvane globalne memorije tokom izvrsenja.
	Nadji pametniji nacin. Sa druge strane dostigli smo **500 GFLOPS**! Trik je u `LOCAL_SIZE = 32`
