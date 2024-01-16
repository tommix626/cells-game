# cells-game
agents evolving on competing resources on a grid.


N group battle: cell-battle.
each grid is a group.

resources on a grid, several species (no offspring, just score)

competition: conflict in grid.
first "hunt down to zero" and then replace.
inital every grid have 10% of their points to conquer.
Use points to conquer, conquering makes more points.



### input: 3+4*3+4 = 19
this grid: 
- resources richness
- count (negative represents conquered by others and positive represents conquered by this(self).)
- tag (0 for self, more positive for more highly socred entity)

- 

surrounding grid: (4/8 direction)
- grid resource richness
- grid count
- grid tag rank


global:
- timer: -100 to 100 from start to end, score is decided at the end.
- current taken spot number. (given by rankings 1-N)
- current total generation speed. (given by rankings 1-N)
- current resources and ranking.


### output: 
- movement (4/8 direction), 
- conquering/reinforcing, drop offsprings on this grid. (points -1, this grid +1 on count.)

points/ taken resources an be negative, at the end, just compare.
reproduction: speed following logistic curve.

## brain

input -> intermediate -> output
NN gene through 8-num hex.

from | to | weight * 8
input intermediate w
intermediate output w
intermediate intermediate w
input output w


32 bit
