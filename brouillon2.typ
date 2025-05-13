#import "template_clean_math_paper.typ": *

#let date = datetime.today().display("[month repr:long] [day], [year]")
#show: template.with(
  title: "Brouillon 2",
  authors: (
    (name: "Léo Burgund"),
    // (name: "Author 1", affiliation-id: 1, orcid: "0000-0000-0000-0000"),
  ),
  affiliations: (
    // (id: 1, name: "Affiliation 1, Address 1"),
    // (id: "*", name: "Corresponding author")
  ),
  date: date,
  heading-color: rgb("#000000"),
  link-color: rgb("#000082"),
  // abstract: [This is my abstract...],
  // keywords: ("First keyword", "Second keyword", "etc."),
  // AMS: ("65M70", "65M12"),
)

#let colMath(x, color) = text(fill: color)[$#x$]
#let ip(x, y) = $lr(angle.l #x, #y angle.r)$

= Problem
Goal:
$
  min_f cal(L) (f).
$
We will study the variations of the loss made by the variations of $S$, with other parameters fixed. Hence we will study
$
  arg min_S cal(L) (S)
$
with
$
  S= X W_Q W_K^top X^top
$

First order approximation:
$
  cal(L) (S+dif S) =cal(L)(S) + ip(G, dif S) + o(norm((dif S)) )
$
with $G = gradient_(S) cal(L)(S)$, and
$
  dif S &= X(W_Q + dif W_Q)(W_K + dif W_K)^top X^top -X W_Q W_K^top X^top \
  &= X W_Q dif W_K^top X^top +X dif W_Q W_K^top X^top +X dif W_Q dif W_K^top X^top \
  &= X(W_Q dif W_K^top +dif W_Q W_K^top ) X^top +o(norm(dif W_Q) dot norm(dif W_K) )\
$<dS1>
#emoji.fire Verify and justify that it is ok to drop $o(norm(dif W_Q) dot norm(dif W_K) )$ (and i there a link with $o(norm(dif S) )$ ?)

We will consider that $dif S =X(W_Q dif W_K^top +dif W_Q W_K^top ) X^top$.

---

We will attempt to resolve the following problem:
$
  arg min_(dif S) ip(G, dif S) space "s.t." norm(dif S) <=gamma
$
with $gamma in RR_+$.

$gamma$ is similar to the learning rate, and constrains $dif S$ to respect the first order approximation.

---

The solution $dif S$ has a norm $norm(dif S) =gamma$ when there exists a $dif S$ such that $ip(G, dif S) <= 0$.

We make the hypothesis that we can always find such a $dif S$.

We then have the following problem:
$
  &arg min_(dif S) ip(G, dif S) space "s.t." norm(dif S)=gamma \
  ( <==>& gamma dot arg min_(dif S) ip(G, dif S) "s.t." norm(dif S) =1)\
$<P>

We have
$
  ip(G, dif S)&= ip(G, X(W_Q dif W_K^top + dif W_Q W_K^top ) X^top) \
  &= ip(X^top G X, W_Q dif W_K^top +dif W_Q W_K^top) space ", let" T=X^top G X \
  &= ip(T, W_Q dif W_K^top) + ip(T, dif W_Q W_K^top) \
  &= ip(dif W_Q, T W_K)+ip(dif W_K, T^top W_Q) \
$

Linear in $dif W_Q, dif W_K$.

The problem now is
$
  arg min_(dif W_Q, dif W_K) ip(dif W_Q, T W_K)+ip(dif W_K, T^top W_Q) space "s.t." norm(X(W_Q dif W_K^top +dif W_Q W_K^top ) X^top)=gamma
$

Hence the "raw directions" of steepest descent to minimize the scalar products are
$
  Delta W_Q^(( 0)) =-T W_K\
  Delta W_K^(( 0)) = -T^top W_Q
$

We define the linear operator
$
  cal(A) (Delta W_Q^((0) ) ,Delta W_K^((0) ) ) :=X(W_Q Delta W_K^top + Delta W_Q W_K^top ) X^top
$

and
$
  dif S^((0) ) :=cal(A) (Delta W_Q^((0) ) ,Delta W_K^((0) ) ), space rho:= norm(dif S^(( 0)))_F
$
We make the hypothesis that $rho !=0$, as we just have to skip the update if it is $0$.

We define
$
  alpha:=gamma / rho
$
and
$
  Delta W_Q:= alpha Delta W_Q^((0) ) , space Delta W_K^((0) ) :=alpha Delta W_K^((0) )
$

We then have
$
  norm(cal(A) (Delta W_Q ,Delta W_K )))_F=alpha rho=gamma
$
so the pair $Delta W_Q, Delta W_K$ have the best minimizing direction for the problem @eq:P, while respecting the norm constraint.

We the have the closed form expressions
$
  rho &= X(W_Q (-T^top W_Q)^top - T W_K W_K^top ) X^top \
  &= -X(W_Q W_Q^top T + T W_K W_K^top ) X^top \
  Delta W_Q^star &= - gamma / rho T W_K \
  Delta W_K^star &= - gamma / rho T^top W_Q\
$

#emoji.fire We made two approximations for this result, the first order approximation and the elimination of the quadratic term $X dif W_Q dif W_K^top X^top$. We should study the consequences it can have, in particular on the validity of the results if $gamma$ or $alpha$ are big, where the quadratic term could become more significant.

Same for the comparaison between $norm(dif W_Q) dot norm(dif W_K)$ and $norm(W_Q) dot norm(dif W_K) + norm(dif W_Q) dot norm(W_K)$.
