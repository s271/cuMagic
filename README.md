cuMagic
=======

cuda visualisation and gamification experiment

To compile this project download CUDA SDK and put the folder into ...\C\src\ directory


This demo was inspired by Escapist article “I hate magic” by Robert Rath (http://www.escapistmagazine.com/articles/view/columns/criticalintel/10302-I-Hate-Magic)  and “continuous game of life” (http://arxiv.org/abs/1111.1567) by Stephan Rafler
This is attempt to show how GPGPU could be used to model magic as different physics.
The demo run at 70 fps at laptop with  GF GTX 670M. With some tradeoffs it can be made run much faster and live enough GPGPU resource for some interesting AI.

The concepts of the game: Player and AI controls elements and laws of magic.
Elements:
Basic elements
The main elements are red and blue. The elements reside on the grey background. Red concentrate on light spots and blue on dark spots. Player control basic element by changing background - moving or jumping gery spiral disk (“form”) Disk can be dragged by mouse or jumpe-moved by mouse click. After sharp move or jump disk disappear and take time to restore (for player only, not AI)
Attack elements.
Attack elements are yellow for player and green for AI. They propagate in viral manner. Player create them by choosing yellow (green for AI) circle and clicking on the game field.
Shield elements
Magenta for player and Jade for AI
Those are kind of vortices which swirl around basic elements and can convert attack elements into itself. They can be harmful for their owner though if used without care. Player create them by choosing magenta (green for AI) ring and clicking on the game field.
Suppressive defence elements:
Orange field suppress green, violet suppress yellow, they also suppress each over. 
Laws:
Low give player and AI some control over conversion of elements.
To invoke low click on the two connected hexagramm. in the column.
Left hexagram and right colors are interacting element, connecting line color is the result of interaction. If right hex is black law “create” resulting element on the boundary of the first element - first element emitting middle element.
If connecting element is black law negate previous low with right and left elements.

Laws and attack/defense elements available to player and ai are stored in files stackHum.txt and stackAI.txt
line “BLACK BLACK YELLOW” correspond to yellow circle in the column - it’s not a low but attack element invoked by clicking on it and place in the field. Same for other 
““BLACK BLACK *”” lines
Available lines for playe (*) mean any color:
* BLACK *
RED ORANGE BLACK
YELLOW  RED BLACK
YELLOW ORANGE BLACK 
ORANGE RED BLACK 
RED MAGNETA BLACK 
BLACK BLACK YELLOW
BLACK BLACK MAGNETA

for AI:
* BLACK *
GREEN VIOLET BLACK 
BLUE VIOLET BLACK
BLUE JADE BLACK
VIOLET BLUE BLACK
GREEN BLUE BLACK
BLACK BLACK JADE
BLACK BLACK GREEN
 
cuph_ai_off.exe - AI turned off, switch between green, yellow, Magenta, Jade is by mouse wheel scroll.
Press 'p' to pause
