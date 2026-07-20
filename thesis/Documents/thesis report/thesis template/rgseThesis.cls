%*******************************************************************************
%  cgBA.cls
%
%  LaTeX-Dokumentenklasse fuer Studien-/Diplomarbeiten der AG Computergraphik.
%
%  (C) 2003-2007 AG Computergraphik <cg@uni-koblenz.de>
%           Universitaet Koblenz-Landau
%  (C) 2019 AG Softwaretechnik <rgse@uni-koblenz.de>
%           Universitaet Koblenz-Landau
%
%  2003-10-09  v1.0  M. Geimer
%  2005-12-22  v1.1  M. Biedermann (neues FB4-Layout fuer DA)
%  2006-01-17  v1.2  M. Biedermann (neues FB4-Layout fuer DA, Einzelarbeit)
%  2007-02-21  v1.3  M. Biedermann (neues Uni-Logo eingebaut)
%  2007-03-21  v1.4  M. Biedermann (Bugfix Schriftgroesse, Dank an Stefan Mueller)
%  2007-04-07  v1.5  M. Biedermann (Englisch-Unterstuetzung wieder hinzugefuegt, 
%									Dank an Florian Schulze und Pascal Sproedt)
%  2007-05-01  v1.6  M. Biedermann (Englisch-Unterstuetzung geaendert (siehe README),
%									kleine Bugfixes)
%  2007-05-15  v1.7  M. Biedermann (Schriftgroesse etc. von Bildunterschriften
%									korrigiert, Dank an Christian Rieder)
%  2008-05-27  v1.8  M. Raspe (Umlautcodierung korrigiert, Dank an Andreas von Arb)
%  2016-07-21  v1.9  N. Klee (neues FB4-Layout fuer Abschlussarbeiten (neues Logo, Genderechtsneutral, vereinfachte Erklaerung), UTF8 Kodierung, Seitenzahl bei "Erklaerung" entfernt)
%  2019-09-04  v2.0  V. Riediger (Vereinfachung und Anpassung an RGSE)
%  2022-11-30  v2.1  V. Riediger (Anpassung neues Uni-Koblenz-Logo)
%  2023-02-03  v2.2  V. Riediger (Neue eideststattliche Erklärung)
%*******************************************************************************


%--- Identification ------------------------------------------------------------

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{rgseThesis}[2023/02/03 v2.2
                   Document class for theses of the
                   Research Group Software Engineering (RGSE)]


%--- Initial code --------------------------------------------------------------

\RequirePackage{ifthen}
\newboolean{@isTimes}
\newboolean{@isPalatino}
\newboolean{@isExtern}
\newboolean{@isMaster}
\newboolean{@isProposal}


%--- Declaration of options ----------------------------------------------------

\DeclareOption{times}{\setboolean{@isTimes}{true}}
\DeclareOption{palatino}{\setboolean{@isPalatino}{true}}
\DeclareOption{twoside}{\PassOptionsToClass{twoside}{report}\PassOptionsToClass{twoside}{article}}
\DeclareOption{intern}{\setboolean{@isExtern}{false}}
\DeclareOption{extern}{\setboolean{@isExtern}{true}}
\DeclareOption{bachelor}{\setboolean{@isMaster}{false}}
\DeclareOption{master}{\setboolean{@isMaster}{true}}
\DeclareOption{proposal}{\setboolean{@isProposal}{true}}


%--- Execution of options ------------------------------------------------------

\ProcessOptions


%--- Package loading -----------------------------------------------------------

\ifthenelse{\boolean{@isProposal}}{%
    \LoadClass[a4paper,11pt]{article}
}{%
    \LoadClass[a4paper,11pt]{report}
}
% Seitenspiegel und Raender
\RequirePackage[a4paper,left=3cm,textwidth=16cm,top=2cm,bottom=2cm,includefoot]{geometry}
\RequirePackage[english,ngerman]{babel}		% support for english an (new) german orthography
\RequirePackage{iflang}	                	% querying current language
\RequirePackage[utf8]{inputenc}				% support native umlauts and special chars
\RequirePackage[T1]{fontenc}				% see above (for PDF)
\RequirePackage{csquotes}                   % multilingual citations
\RequirePackage[export]{adjustbox}          % http://ctan.org/pkg/adjustbox
\RequirePackage{graphicx}					% for images
\RequirePackage{tabularx}					% flexible tables
\RequirePackage[hang,small,bf]{caption}		% captions with options
\RequirePackage{amsfonts}					%
\RequirePackage{amssymb}					% Math stuff
\RequirePackage{amsmath}					%
\RequirePackage[super]{nth}
\RequirePackage{setspace} %Zeilenabstand ändern
        %       \singlespacing %einfach
        %       \onehalfspacing %1,5-fach
        %       \doublespacing %doppelt
        %       \linespread{1.25}
        %       \baselineskip15pt
\RequirePackage{prettyref}  % Huebsche Referenzen
\RequirePackage[pdftex,plainpages=false]{hyperref} % Links in PDF-LaTeX 
\newrefformat{fig}{\hyperref[{#1}]{Figure~\ref*{#1}}}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%						% Default: Computer Modern (TeX-Standard)
\ifthenelse{\boolean{@isTimes}}{%
         \RequirePackage{times}					% Schriftart Times New Roman
      }{}
\ifthenelse{\boolean{@isPalatino}}{%
         \RequirePackage{palatino}			% Schriftart Book Antiqua/Palatino
      }{}      
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%--- Code ----------------------------------------------------------------------

\def\@supervisor{}
\def\@supervisorGender{}
\newcommand{\supervisor}[2]{\def\@supervisorGender{#1}\def\@supervisor{#2}}

\def\@supervisorInfo{}
\newcommand{\supervisorInfo}[1]{\def\@supervisorInfo{#1}}

\def\@secondSupervisor{}
\def\@secondSupervisorGender{}
\newcommand{\secondSupervisor}[2]{\def\@secondSupervisorGender{#1}\def\@secondSupervisor{#2}}

\def\@secondSupervisorInfo{}
\newcommand{\secondSupervisorInfo}[1]{\def\@secondSupervisorInfo{#1}}

\def\@externLogo{}
\newcommand{\externLogo}[2]{\def\@externLogo{\includegraphics[width={#1},valign=t]{{#2}}}}

\def\@externName{}
\newcommand{\externName}[1]{\def\@externName{#1}}

\def\@programOfStudy{}
\newcommand{\programOfStudy}[1]{\def\@programOfStudy{#1}}


\newcommand{\abgabeDE}{%
   \ifcase\month\or
      Januar\or Februar\or M{\"a}rz\or April\or Mai\or Juni\or
      Juli\or August\or September\or Oktober\or November\or Dezember
   \fi\ \number\year
}
\newcommand{\abgabeEN}{%
   \ifcase\month\or
      January\or February\or March\or April\or May\or June\or
      July\or August\or September\or October\or November\or December
   \fi\ \number\year
}



%%% new title page
\renewcommand{\maketitle}{
    \pagestyle{empty}
    \IfLanguageName{english}{%
        \ifthenelse{\boolean{@isExtern}}{%
            % with external logo
            \includegraphics[width=6cm,valign=t]{logos/UK_FB4_Logo_eng_Schutzraum} \hfill \@externLogo\par
            \textsf{Faculty 4: Computer Science} \hfill \textsf{\@externName}
        }{%
            % Uni logo only
            \begin{center}
            \includegraphics[width=7.5cm]{logos/UK_FB4_Logo_eng_Schutzraum}\par
            \textsf{Faculty 4: Computer Science}
            \end{center}
        }
      }{%
        \ifthenelse{\boolean{@isExtern}}{%
            % with external logo
            \includegraphics[width=6cm,valign=t]{logos/UK_FB4_Logo_Schutzraum} \hfill \@externLogo\par
            \textsf{Fachbereich 4: Informatik} \hfill \textsf{\@externName}
        }{%
            % Uni logo only
            \begin{center}
            \includegraphics[width=7.5cm]{logos/UK_FB4_Logo_Schutzraum}\par
            \textsf{Fachbereich 4: Informatik}
            \end{center}
        }%
    }

    \vfill      

    \begin{center}
        \IfLanguageName{english}{%
            \begin{otherlanguage}{english}
                \huge\begin{spacing}{1.25}
                \textbf{\@title}
                \end{spacing}
            \end{otherlanguage}
        }{%
            \begin{otherlanguage}{ngerman}
                \huge\begin{spacing}{1.25}
                \textbf{\@title}
                \end{spacing}
            \end{otherlanguage}
        }
        \bigskip
        \ifthenelse{\boolean{@isProposal}}{%
            \huge
            \IfLanguageName{english}{%
                Proposal for a 
                \ifthenelse{\boolean{@isMaster}}{Master Thesis}{Bachelor Thesis}
            }{%
                Exposee für eine
                \ifthenelse{\boolean{@isMaster}}{Masterarbeit}{Bachelorarbeit}
            }
            \medskip

            \IfLanguageName{english}{%
                \large submitted by\\[2ex]\Large\@author
            }{%
                \large vorgelegt von\\[2ex]\Large\@author
            }
            
            \bigskip

            \normalsize
            \today
        }{%
            {\huge \ifthenelse{\boolean{@isMaster}}{Masterarbeit}{Bachelorarbeit}}

            zur Erlangung des Grades 
            \ifthenelse{\boolean{@isMaster}}{Master of Science (M.Sc.)}{Bachelor of Science (B.Sc.)}\\
            im Studiengang \ifthenelse{\equal{\@programOfStudy}{}}{Informatik}{\@programOfStudy}

            \medskip

            {\large vorgelegt von\\[2ex]\Large\@author}
        }
    \end{center}	

    \vfill

    \vfill      

    \begin{tabular}{@{}ll}
     \ifthenelse{\boolean{@isProposal}}{%
         \IfLanguageName{english}{Supervisor:}{\ifthenelse{\equal{\@supervisorGender}{w}}{Betreuerin:}{Betreuer:}}
     }{\ifthenelse{\equal{\@supervisorGender}{w}}{Erstgutachterin:}{Erstgutachter:}}  
     & \ifthenelse{\equal{\@supervisor}{}}{Prof. Dr. Jan J{\"u}rjens}{\@supervisor} \\
     & \ifthenelse{\equal{\@supervisorInfo}{}}{Institut f{\"u}r Softwaretechnik}{\@supervisorInfo} \\[1ex]

     \ifthenelse{\equal{\@secondSupervisor}{}}{}{%
         \ifthenelse{\boolean{@isProposal}}{\IfLanguageName{english}{\nth{2} Supervisor:}{\ifthenelse{\equal{\@secondSupervisorGender}{w}}{Zweitbetreuerin:}{Zweitbetreuer:}}}
         {\ifthenelse{\equal{\@secondSupervisorGender}{w}}{Zweitgutachterin:}{Zweitgutachter:}}  
         & \@secondSupervisor \\
         & \@secondSupervisorInfo \\
     }
    \end{tabular}

    \ifthenelse{\boolean{@isProposal}}{%
      % empty for proposal
    }{%
        \vspace{10ex}

        Koblenz, im \abgabeDE
    }

    \cleardoublepage

    \pagestyle{plain}
	
    \ifthenelse{\boolean{@isProposal}}{%
        % no declaration in proposal
    }{%
        %%%  declaration  %%%
        \begin{otherlanguage}{ngerman}
    	\section*{Eidesstattliche Erkl{\"a}rung:}
    
        \medskip
	
Hiermit best{\"a}tige ich, dass die vorliegende Arbeit von mir selbst{\"a}ndig verfasst wurde und ich keine anderen als die angegebenen Hilfsmittel – insbesondere keine im Quellenverzeichnis nicht benannten Internet–Quellen – benutzt habe und die Arbeit von mir vorher nicht in einem anderen Pr{\"u}fungsverfahren eingereicht wurde.\\

        \medskip
	
    	\begin{flushleft}
    	\begin{tabularx}{\linewidth}{@{}lcc}
        & Ja & Nein \\[2ex]
    	Mit der Einstellung der Arbeit in die Bibliothek bin ich einverstanden. & $\square$ & $\square$ \\
    	\end{tabularx}
    	\end{flushleft}

    	\vspace{3cm}
	
    	\dotfill
    
    	{\small(Ort, Datum) \hfill (Unterschrift)}
        \end{otherlanguage}
       \cleardoublepage
    } % ifthenelse
} % maketitle

\usepackage[
backend=biber,
style=ieee,
style=numeric,
maxbibnames=99
]{biblatex}

%Entrytype definition and driver for standards (ISO, ECSS, W3C ...)
\DeclareDatamodelEntrytypes{standard}
\DeclareDatamodelEntryfields[standard]{organization,institution,number,type}
\DeclareBibliographyDriver{standard}{%
      \usebibmacro{bibindex}%
      \usebibmacro{begentry}%
      \usebibmacro{author}%
      \setunit{\labelnamepunct}\newblock
      \usebibmacro{title}%
      \newunit\newblock
      \printfield[parens]{type}
      \newunit\newblock
      \printfield{number}%
      \setunit{\addspace}\newblock
      \printtext[parens]{\printlist{organization}%
      \setunit{\addspace}\newblock
      \printlist{institution}}%
      %\printfield[parens]{type}%
      \newunit\newblock
      \usebibmacro{location+date}%
      \newunit\newblock
      \iftoggle{bbx:url}
      {\usebibmacro{url+urldate}}
      {}%
      \newunit\newblock
      \usebibmacro{addendum+pubstate}%
      \setunit{\bibpagerefpunct}\newblock
      \usebibmacro{pageref}%
      \newunit\newblock
      \usebibmacro{related}%
      \usebibmacro{finentry}}

% separate paragraphs by vertical space instead of indents
\parindent0cm
\parskip2ex plus1ex minus0.5ex
