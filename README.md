## This is an implementation of the system proposed in the paper titled
# Computer Vision-Based Self-Inflicted Violence Detection in High-Rise Environments using Deep Learning
## presented at the 2nd International Conference on Power Engineering and Intelligent Systems(PEIS2024)

### The architectural view of the proposed system
> ![Architecture Suicide](https://github.com/Priykrit/Computer-Vision-Based-Self-Inflicted-Violence-Detection-in-High-Rise-Environments-using-Deep-Learn/assets/98400044/9ab655b5-c456-48c0-a61b-cba4622207d1)

Scene from processed video feed localizing the persons and their score w.r.t. Hazardous boundary. Also, localized the boundary from which self-inflicted violence may occur. The person's movement score at the initial stages is only because of his movement and increases with time and movements
> ![1](https://github.com/Priykrit/Computer-Vision-Based-Self-Inflicted-Violence-Detection-in-High-Rise-Environments-using-Deep-Learn/assets/98400044/0dd371ae-4d9a-43e4-b761-259f2b614567)

Feed indicating the score and color change w.r.t. Fig. 1 shows when a person performs suspicious behavior at a high-risk area for a specific time. When the person approaches the boundary of the hazardous location, the score starts to increase and becomes 0.14 when he reaches close to boundary
> ![2](https://github.com/Priykrit/Computer-Vision-Based-Self-Inflicted-Violence-Detection-in-High-Rise-Environments-using-Deep-Learn/assets/98400044/5fb59adb-21c1-40b6-925e-0040fe2c9f21)

The score rapidly increased and became 0.59 due to randomness in movement
> ![3](https://github.com/Priykrit/Computer-Vision-Based-Self-Inflicted-Violence-Detection-in-High-Rise-Environments-using-Deep-Learn/assets/98400044/49298213-7662-46aa-9cb6-a4ee1447c933)

Feed indicating score and color change w.r.t. Fig 1, 2, and 3 when a person stands on the boundary of a hazardous location with notification of the person's ID and amount of time spent on that place. When a person climbs the boundary, it can be seen that now the score is 0.7
> ![4](https://github.com/Priykrit/Computer-Vision-Based-Self-Inflicted-Violence-Detection-in-High-Rise-Environments-using-Deep-Learn/assets/98400044/923697da-6b30-4d93-b44c-c5a46f227465)

Feed indicating score, time, and color change w.r.t. Fig. 1, 2, 3, and 4 when the person stands again on top of the boundary wall, and notifications concerning changed information are displayed. This score will keep increasing with time to a certain threshold if the person remains in the frame 
> ![5](https://github.com/Priykrit/Computer-Vision-Based-Self-Inflicted-Violence-Detection-in-High-Rise-Environments-using-Deep-Learn/assets/98400044/1dceb8f1-b43d-496e-859d-b57041a2e690)

### [Click to watch video](https://youtu.be/XOl7wWm40Wo).
### To Run code please run *pip install -r requirements.txt* in terminal first to install all requirments
