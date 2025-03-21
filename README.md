# Harmonic Content of String Networks

The goal of this project is to analyze the frequency content of audio samples taken in a lab at Monmouth University.  Specifically, we are analyzing the harmonic content of 

1.  single steel guitar strings,
2.  2 concatenated steel guitar strings,
3.  2 twisted steel guitar strings, and
4.  3 steel guitar strings connected in a Y shape.

Why are we doing this?  Like all great things, it started with math.  If you'd like to see the outline of the theoretical models, scroll down!  If you'd like to get an explanation of how to use this repository, then you're in the right place.

## The organization

This repository is organized into two branches:

1. Python files
2. Audio sample files

#### Python files

*description needed*

#### Audio samples

In the "Audio Samples" branch, you will find .zip folders containing the audio samples in .wav format.  The naming convention for the .zip folders is organized by number of strings and gauge (string thickness).  For example, "2S8" means 2 concatenated 8-gauge strings, and "1S16" means a single 16-gauge string. 

## Mathematical Models

#### One string

The simplest mathematical model of a vibrating string which is pinned down at both ends is known as the *wave equation*.  For a string of length $L$ with some initial condition $f(x)$ (think of it as a snapshot at the beginning of the oscillation), the governing partial differential equation with boundary and initial conditions looks like this:

```math
\begin{cases}
v_{tt} = c^2 v_{xx}\\
v(0,t) = 0 = v(L,t)\\
v(x,0) = f(x)
\end{cases}
```

where $v(x,t)$ is the displacement of the string off of equilibrium at position $x$ along the string's length and time $t$, and $c$ is the *wave speed* that can be calculated from the *tension* and *linear density* (i.e. thickness) of the taut string.  It turns out that this differential equation is *separable*, which means that we can consider the time and space components separately.  Doing some math, we find that we actually want to solve the following spatial problem:

```math
\begin{cases}
- u'' = k^2 u\\
u(0) = 0 = u(L)
\end{cases}
```

where $k$ is an unknown constant.  The solution to this differential equation turns out to be

```math
u(x) = A\cos(kx) + B\sin(kx)
```

and, evaluating the boundary values yields

```math
A = 0 \quad \text{and} \quad B\sin(kL) = 0.
```

In particular, the second condition tells us that 

```math
k = \frac{n \pi}{L}, \quad n = 1, 2, 3, \ldots
```

and these are the predicted *harmonics* of a vibrating string!  From here, we made the problem more complicated by imagining that two strings were pinned down and concatenated end-to-end at a central node in a way that allowed them to freely oscillate, as in the following crude diagram:

<pre>
  str 1      node      str 2
o-------------o-----------------o
|                               |
</pre>

#### Two strings

We will consider two strings concatenated (as shown above) with lengths $\ell_1$ and $\ell_2$.  The mathematical model of this scenario is known as a *quantum graph* (in short, a metric graph with a differential operator acting on each edge).  The mathematical description of the connection is crucial: we will demand that the strings remained connected (known as a *continuity* requirement) and that, effectively, the strings will not kink at the central node.  When written as a boundary value problem (with the left endpoint labeled as $\ell_1$, the central node labeled as $0$, and the right endpoint labeled as $\ell_2$), we have

```math
\begin{cases}
- u'' = k^2 u\\
u_1(\ell_1) = 0 = u_2(\ell_2)\\
u_1\prime(0) + u_2\prime(0) = 0\\
u_1(0) = u_2(0)
\end{cases}
```

Similarly to above, solving this yields harmonics of the form

```math
k = \frac{n \pi}{\ell_1 + \ell_2}, \quad n = 1, 2, 3, \ldots
```

which means that, when plucked, the 2-string system should sound exactly like one string whose length is the sum of the 2 string lengths!  In other words, our concatenated 2-string system is (theoretically) just 1 long string in disguise!
