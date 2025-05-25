#import "template_clean_math_paper.typ": *

#let date = datetime.today().display("[month repr:long] [day], [year]")
#show: template.with(
  title: "Brouillon 4",
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

= Nomenclature
== Dimensions
- $b$ Mini-batch size
- $d_e$ Embedding dimension
- $d_s$ Sequence length
- $d_k$ Query/Keys dimension
- $d_v$ Value dimension
- $h$ Number of heads

We make the hypothesis that $d_k < d_e < d_s$.

== Matrix operations in a self-attention block
// TODO: Check if h factor is numerator or denominator
In the case of multi head attention, for each head $i = 1,...,h$, we have:
- Input $X in RR^(d_s times d_e)$
- $W_Q_i in RR^(d_e times d_k / colMath(h, #red)), Q_i:= X W_Q_i in RR^(d_s times d_k / colMath(h, #red))$
- $W_K_i in RR^(d_e times d_k / colMath(h, #red)), K_i:= X W_K_i in RR^(d_s times d_k / colMath(h, #red))$
- $S_i := (Q_i K_i^top ) / sqrt(d_k / colMath(h, #red)) in RR^(d_s times d_s)$
- $A_i := "softmax"_"row" (S)$
- $W_V_i in RR^(d_e times d_v / colMath(h, #red)), V_i:= X W_V_i in RR^(d_s times d_v / colMath(h, #red))$
- $H_i := A_i V_i in RR^(d_s times d_v / colMath(h, #red))$, $H=[H_1,...,H_h] in RR^(d_s times d_v)$
- $W_O in RR^(d_v times d_e)$
- Output $Y:= H W_O + X in RR^(d_s times d_e)$

For now, we study the case $h=1$.

#emoji.fire We omit the $1 / sqrt(d_k)$ scaling for the $S$ matrix, it can cause problem with growing, so for growing we will make it a learnable parameter (and initialize it at $1 / sqrt(d_k_"initial")$ ?). Or maybe scale with $(sqrt(d_k) / sqrt(d_k +p) )$? But need to maintain the same output for the model?

= Problem
Goal:
$
  min_f cal(L) (f).
$
We will study the variations of the loss made by the variations of $S$, with other parameters fixed. Hence we will study
$
  arg min_S cal(L) (S)
$
Let $G:= gradient_(S) cal(L)(S)$.

We have
$
  "rk"(G) <= d_s, space "rk"(S) <=d_k
$


We have the first order approximation, with the introduction of $gamma in RR_+^*$, similar to a step size:
$
  cal(L) (S+ gamma dif S) =cal(L)(S) + gamma ip(G, dif S)_F + o(norm((dif S))_F )
$

We consider the problem
$
  arg min_(dif S) ip(G, dif S)_F "s.t." norm(dif S)_F <= gamma
$


Let $Z= W_Q W_K^top$, with $"rk"(Z)<= d_k$ , we then have
$
  S&= X W_Q W_K^top X^top\
  &= X Z X^top \
$
and
$
  dif S &= X ( W_Q_(+1)W_K_(+1))X^top - X W_Q W_K^top X^top \
  &= X (Z + dif Z) X^top -X Z X^top \
  &= X dif Z X^top \
$

Moreover,
$
  "rk"(dif S) ="rk"(X dif Z X^top ) = "rk"(dif Z) <= d_k < d_e < d_s
$


Hence, the problem becomes
$
  dif Z^star &=arg min_(dif Z) ip(G, X dif Z X^top)_F "s.t." norm(X dif Z X^top)_F <= gamma\
  &= - arg max_(dif Z) ip(G, X dif Z X^top)_F "s.t." norm(X dif Z X^top)_F <= gamma \
  &= - arg max_(dif Z) ip(G, X dif Z X^top)_F "s.t." norm(X dif Z X^top)_F = gamma space (*) \
  &= - gamma arg max_(dif Z) ip(G, X dif Z X^top)_F "s.t." norm(X dif Z X^top)_F = 1 \
  &= - gamma / alpha arg min_(dif Z) norm(G - X dif Z X^top)_F^2\
$

$(*)$ We make the hypothesis that we can always find a $ip(G, X dif Z X^top)_F >0$.

#emoji.fire Justification for $alpha$, $alpha= norm(?)_F$

Let $lambda := gamma / alpha$, $P:= arg min_(dif Z) norm(G - X dif Z X^top)^2_F$.

We will first search $P$, then $lambda$ with a line search.

== Find $P$
Let
$
  f(P)=norm(G - X P X^top)_F^2.
$
$P |-> X P X^top$ is linear, $P |-> G-X P X^top$ is affine, and $A |-> norm(A)^2_F$ is convex. $f$ is a composition of those functions, hence is convex.

We have
$
  gradient_(P) f = -2 X^top (G - X P X^top )X
$
so
$
  gradient_(P) f = 0 <==> X^top X P X^top X = X^top G X
$

=== $X$ full column rank (#emoji.fire true in practice?)
Under the hypothesis that $X$ has full column ($d_e$) rank, $X^top X$ is invertible, we have the pseudoinverse
$
  X^+ =(X^top X)^(-1) X^top
$
and the solution

$
  P^star &= (X^top X)^(-1) X^top G X (X^top X)^(-1) \
  &= (X^top X )^+ X^top G X (X^top X )^+ \
  &= X^+ G (X^+ )^top \
$
#emoji.fire Which formula to implement for $P^star$ ? Numerical stability?

=== $X^top X$ not invertible
Let
$
  cal(A) := P |-> X^top X P X^top X
$
If $X^top X$ is not invertible, $X^top X$ is not injective, and under the hypothesis that $X^top X !=0$,
$
  X^top X N X^top X=0 &<==> X N X^top =0
$
hence
$
  ker(cal(A)) ={N in RR^(d_e times d_e) bar X N X^top =0 }
$

Hence any solution $P_0$ of $X^top X P X^top X = X^top G X$ can be changed by $N in ker(cal(A))$.

$
  P=P_0+N => X^top X(P_0+N) X^top X=X^top X P_0 X^top X
$
We can always take $N=0$.

#emoji.fire Can we take $P_0=X^+ G(X^+ )^top$ ?


== Batch
To account for the batch, there are several ways to average:
$
  EE_X [(X^top X)^+ X^top G X (X^top X )^+ ] = EE_X [X^+ G ( X^+ )^top ] \
  EE_X [X^top X]^+ EE_X [X^top G X] EE_X [X^top X]^+ \
  EE_X [X]^+ EE_X [G] (EE_X [X]^+ )^top \
  EE_X [X^top X]^(-1) EE_X [X^top G X] EE_X [X^top X]^(-1) \
$
#emoji.fire Which one to take?

When $d_s >> d_e$, we see with experiments:
$
  EE_X [X^top X]^+ EE_X [X^top G X]EE_X [X^top X]^+ -> EE_X [(X^top X)^+ X^top G X (X^top X )^+ ]
$
With the left member being cheaper to compute

#emoji.fire Test full pinv VS (if possible inv else pinv)


== Line search
Do a line search to find $lambda$
=== "Normal" way
Line search on
$
  cal(L) (X space "SVD"_(d_k + p)(W_Q^(t) W_K^t^top - lambda P)X^top )
$
=== Testing fast way
Lose the rank constraint, but lose the SVD so faster. Get a good approximation of $lambda$ ?
$
  cal(L) (X ( W_Q^(t)W_K^t^top -lambda P)X^top ) &= cal(L) (X W_Q^t W_K^t^top X^top - lambda X P X^top ) \
$

