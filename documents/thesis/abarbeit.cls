\NeedsTeXFormat{LaTeX2e}[1994/06/01]
\ProvidesClass{abarbeit}[2024/05/22 abarbeit]
%% Author: Erich Schubert
%% Version 0.1.1, 2024/05/22
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\RequirePackage{ifthen}
\RequirePackage{iftex}
\newif\ifTUDO@english\TUDO@englishtrue % Default to English
\newif\ifTUDO@fira\TUDO@firafalse % Default to Linux Libertine
\newif\ifTUDO@onehalfspacing\TUDO@onehalfspacingfalse % No extra spacing
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% OPTIONEN (zzgl. KOMA-Script-Optionen, https://komascript.de/~mkohm/scrguide.pdf):
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\RequirePackage{xkeyval}
% "german" to write in German
\DeclareOptionX{german}{\TUDO@englishfalse}
% "deutsch" to write in German
\DeclareOptionX{deutsch}{\TUDO@englishfalse}
% "english" to write in English
\DeclareOptionX{english}{\TUDO@englishtrue}
% "libertine" for Linux Libertine font (default)
\DeclareOptionX{libertine}{\TUDO@firafalse}
% "fira" for Fira fonts
\DeclareOptionX{fira}{\TUDO@firatrue}
% "onehalfspacing" to use one-half spacing
\DeclareOptionX{onehalfspacing}{\TUDO@onehalfspacingtrue}
% pass other options to Koma script
\DeclareOptionX*{\PassOptionsToClass{\CurrentOption}{scrbook}}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\ProcessOptionsX\relax
% Load Koma script book class, with some default values
\LoadClass[a4paper,12pt,parskip=half-,DIV=13,BCOR=10mm,bibliography=totoc,listof=totoc,headings=small]{scrbook}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\usepackage[utf8]{inputenc} % default by now anyway, but csquotes complains otherwise
\usepackage[T1]{fontenc}
%%% Babel und csquotes laden
\ifTUDO@english
  \usepackage[german,main=english]{babel}
  \usepackage[autostyle=true]{csquotes}
\else
  \usepackage[english,main=german]{babel}
  \usepackage[autostyle=true,german=quotes]{csquotes}
\fi

%%% Zeilenabstand
\ifTUDO@onehalfspacing
  \usepackage[onehalfspacing]{setspace} % Nur wenn der Gutachter darauf besteht.
\fi

%%% Schrift: Linux Libertine + Inconsolata, alternativ alles Fira
\ifTUDO@fira
  % Alternative Schrift: alles in Fira
  \usepackage[sfdefault,medium,lining,scaled=.9]{FiraSans}
  \usepackage[medium,lining,scaled=.9]{FiraMono}
  \usepackage[italic,basic,noparenthesis]{mathastext}
  \ifpdftex
    \usepackage{newtxsf} % pdflatex version
  \else % LuaLaTeX / XeLaTeX
    \usepackage[math-style=TeX,warnings-off={mathtools-colon,mathtools-overbracket}]{unicode-math}
    \setmathfont{TeX Gyre Pagella Math}[Scale=MatchLowercase]
    \setmathfont{Fira Sans}[range={up/{num,latin,Latin,greek,Greek},%
          \mathexclam,\mathplus,\pm,\div,\minus,\mathpercent,\mathampersand,%
          \mathquestion,\mathatsign,\increment,\less,\equal,\greater,\ne,\leq,%
          \geq,\matheth,\ell,\partial},%
          Script=Latin,script-features={}, sscript-features={}]
    \setmathfont{Fira Sans Italic}[range={it/{latin,Latin,greek,Greek}}, Script=Latin, script-features={}, sscript-features={}]
  \fi
\else % Default: Fira nur für Titelseite, sonst Linux Libertine + Inconsolata
  \usepackage[medium,lining,scaled=.9]{FiraSans} % Für Titelseite
  \usepackage{libertine}
  \usepackage[libertine]{newtxmath}
  \ifpdftex
    \usepackage[narrow]{inconsolata}
  \else % LuaLaTeX / XeLaTeX version
    \usepackage{fontspec}\setmonofont{inconsolata}[FakeStretch=0.9]
  \fi
\fi

\usepackage{graphicx}
\usepackage[svgnames]{xcolor}
\usepackage{url}
\definecolor{LinkColor}{RGB}{92,128,17} % Abgedunkelte TU-Farbe
\usepackage[colorlinks=true,linkcolor=LinkColor,urlcolor=LinkColor,citecolor=LinkColor,anchorcolor=LinkColor,pdfborder={0 0 0}]{hyperref} % ohne Rand
\usepackage{booktabs}
\usepackage{enumitem}\setlist{nosep}
\usepackage{amsmath,mathtools}

%%%% Mehr line breaks in URLs erlauben
\AtBeginDocument{
\mathchardef\UrlBigBreakPenalty=100\relax\def\UrlBigBreaks{\do\:\do@url@hyp\do\-\do\/}
\ifcsname biburlsetup\endcsname % falls kein biblatex
\let\@oldbiburlsetup\biburlsetup\renewcommand*{\biburlsetup}{\@oldbiburlsetup\mathchardef\UrlBigBreakPenalty=100\relax\def\UrlBigBreaks{\do\:\do@url@hyp\do\-\do\/}}%
\fi
}

\clubpenalty=3000 \widowpenalty=3000 \displaywidowpenalty=3000 % Optional: Witwenzeilen und Waisenzeilen vermeiden. Sie können diese Zeile auskommentieren, wenn das Layout dann schöner ist!
\raggedbottom % keine großen Abstände um seiten vertikal voll zu machen, macht freien Platz sichtbarer

%%%% Table of contents: keine farbigen links, kein one-half spacing
\addtocontents{toc}{\protect\hypersetup{linkcolor=black}}

\ifcsname setstretch\endcsname
\RequirePackage{xpatch}
%\xapptocmd{\frontmatter}{\setstretch{1}\KOMAoptions{DIV=current}}{}{} % Vorwort might be fine in 1.5
\xapptocmd{\mainmatter}{\setstretch{1.5}\KOMAoptions{DIV=current}}{}{}
\xapptocmd{\nackmatter}{\setstretch{1}\KOMAoptions{DIV=current}}{}{}
\fi

%%%% Titelseite
\RequirePackage[absolute]{textpos}
\DeclareRobustCommand{\maketitle}{%
\ifcsname erstgutachterIn\endcsname\else\errmessage{\\erstgutachterIn nicht gesetzt}\fi
\ifcsname zweitgutachterIn\endcsname\else\errmessage{\\zweitgutachterIn nicht gesetzt}\fi
\ifcsname arbeitsgruppe\endcsname\else\errmessage{\\arbeitsgruppe nicht gesetzt}\fi
\bgroup
\pagenumbering{alph}
\begin{titlepage}\firalight
\ifcsname hypersetup\endcsname\hypersetup{linkcolor=black}\fi% Links nicht hervorheben
\ifcsname setstretch\endcsname\setstretch{1}\fi% Kein one-half-spacing
\KOMAoptions{DIV=current}%
\begin{textblock*}{57.6mm}(23.7mm,11.6mm)%
\includegraphics[width=57.6mm]{pictures/tu-dortmund-logo.pdf}%
\end{textblock*}
%\tikz[remember picture,overlay]{\node[anchor=north west,xshift=23.7mm,yshift=-11.6mm,inner sep=0pt] at (current page.north west){\includegraphics[width=57.6mm]{bilder/tu-dortmund-logo.pdf}}}%

\vspace*{4cm}
\bgroup
\centering
\makeatletter
{\Large \@subject} \\
\vspace*{1cm}
{\bfseries\Large \@title} \\
\vspace*{1cm}
\@author\\
%\matrikelnummer{}\\
\vspace*{5mm}
\@date
\par
\makeatother
\egroup

\vfill
\bgroup
\raggedright
Begutachtung: \\
\erstgutachterIn{} \\
\zweitgutachterIn{}
\egroup

\vspace*{2.5cm}
\begin{minipage}[b]{.5\linewidth}\raggedright
\arbeitsgruppe{}
\end{minipage}
\hfill
% OPTIONAL bei Kooperation mit anderen Lehrstuehlen, falls definiert
\ifcsname kooperation\endcsname
\begin{minipage}[b]{.49\linewidth}\raggedleft
\kooperation{}%
\end{minipage}
\fi
\end{titlepage}
\egroup
\KOMAoptions{DIV=current} % recompute with previous font
}

\endinput
