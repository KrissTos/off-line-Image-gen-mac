





# Eyes Direction LoRA Flux 2 Klein 9B v1

**Trigger word:** `change the eyes to match the reference dot direction`

LoRA trained to change where the eyes are looking at relative to the camera.

This lora works with everything that resembles a human eye. It takes the color and the pupil shape from the supplied image.
Works with realisitc images, cg images, comics, anime, paintings, drawings... you name it. 
Whatever thing that resembles an eye, it will try to move it keeping the visual style.

Has some caveats though:
It doesn't work very well with animals, this lora never saw animal images in the training, is not for that. If you stick a cat in there, can't asure you it will work. But worth a try.
It doesn't work very well when the eye is very small in frame or extremely big filling almost the entire frame. But again, maybe worth a try to see what you can get.
It doesn't like very much eyes with different colors/pupil sizes.. so with David Bowie doesn't work.


---

This LoRA uses a red dot reference image, so I created a Eyes Direction Control node to make easier to specify where the eyes have to look at.

If you don't want to use the node, just recreate the image in Photosop, come on, don't be lazy.

Download the Eyes Direction Control node from here:

- [Eyes Direction Control node](https://github.com/eric-venti-seeds/Eyes_Direction_Lora_Control)

<div align="center">

<img src="assets/Eyes_direction_node.gif" width="350"/>

</div>

# Eyes Direction LoRA Flux 2 Klein 9B v1

How it works:

The eyes will always look at the red dot. 
If is in the line, the eyes will "kind of" look to the border of the image. You move it more to the right, the eyes will look outside the image.
(Yes, like when you prompt "look to the left outside the image" and does whatever it wants)

<div align="center">
<img src="https://cdn-uploads.huggingface.co/production/uploads/682506987fd6f758ef6c99a4/kDD7WN8UxFA4pZloaTK2q.png" width="400"/>
</div>

A red dot in the center of the canvas will always make the eyes look to camera, regardless of the body position. 
It is trained with the correct eyes limits, so if you put the point in a "impossible position" then probably the model will move the pupil in a random direction:

<div align="center">
<img src="assets/face_turning.gif" width="1080"/>
</div>

Use the red dot inside the border when you have extreme head positions like totally from one side:

<div align="center">
<img src="assets/face_side.gif" width="550"/>
</div>


## Examples

Works with humans:




<p align="center">
  <img src="assets/selfie.gif" width="1080"/><br/>
  <img src="assets/goodbadugly.gif" width="800"/><br/>
  <img src="assets/emma.gif" width="800"/><br/>
  <img src="assets/faces.gif" width="800"/><br/>
  <img src="assets/faceangles.gif" width="800"/>
  <img src="assets/rihanna.gif" width="800"/><br/>
</p>




And with whatever that resembles an eye in any style:

<p align="center">
  <img src="assets/djones.gif" width="800"/><br/>
  <img src="assets/bulma.gif" width="800"/><br/>
  <img src="assets/cggirl.gif" width="800"/><br/>
  <img src="assets/db.gif" width="800"/><br/>
  <img src="assets/mouse.gif" width="800"/><br/>
  <img src="assets/mug_eyes.gif" width="800"/><br/>
  <img src="assets/princesses.gif" width="800"/><br/>
  <img src="assets/simpsons.gif" width="800"/>
</p>



## How to use

The trigger sentence is:

`change the eyes to match the reference dot direction`

You have the workflow inside the workflow folder.

If you have a non-realistic style to change the eyes, works better if you add an extra sentence with the style the image has.

From my tests, for humans and realistic style works well between Strength 0.5-1.
For anime and non realistic styles sometimes has to be pushed to 1.25-1.5.



Some of the training data was extracted from the Columbia Gaze Dataset here:
https://ceal.cs.columbia.edu/columbiagaze/

And the publication Gaze Locking: Passive Eye Contact Detection for Human–Object Interaction from Brian A. Smith, Qi Yin, Steven K. Feiner, Shree K. Nayar is here:
https://dl.acm.org/doi/10.1145/2501988.2501994?cid=99659562550


Hope you like it and let me know if it works correctly!

Eric Venti.
