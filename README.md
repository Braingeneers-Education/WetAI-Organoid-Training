# WetAI: Collaborative Neuroscience

[![ssec](https://img.shields.io/badge/SSEC-Project-purple?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA0AAAAOCAQAAABedl5ZAAAACXBIWXMAAAHKAAABygHMtnUxAAAAGXRFWHRTb2Z0d2FyZQB3d3cuaW5rc2NhcGUub3Jnm+48GgAAAMNJREFUGBltwcEqwwEcAOAfc1F2sNsOTqSlNUopSv5jW1YzHHYY/6YtLa1Jy4mbl3Bz8QIeyKM4fMaUxr4vZnEpjWnmLMSYCysxTcddhF25+EvJia5hhCudULAePyRalvUteXIfBgYxJufRuaKuprKsbDjVUrUj40FNQ11PTzEmrCmrevPhRcVQai8m1PRVvOPZgX2JttWYsGhD3atbHWcyUqX4oqDtJkJiJHUYv+R1JbaNHJmP/+Q1HLu2GbNoSm3Ft0+Y1YMdPSTSwQAAAABJRU5ErkJggg==&style=plastic)](https://escience.washington.edu/wetai/)
[![BSD License](https://badgen.net/badge/license/BSD-3-Clause/blue)](LICENSE)
[![DOI](https://zenodo.org/badge/691200258.svg)](https://zenodo.org/badge/latestdoi/691200258)

## Assignment: Organoid-Training

The first half of the final project where you will be teaching a live mouse organoid to play a game on a computer! This is a group assignment so work in teams of 4-5 people, you all should have access to the same repository and will be able to share code via it. You can work on the homework in individual codespaces which will make sure you get used committing and pushing/pulling frequently so your version of the assignment stays up to date with everyone else's, or you can use the "live share" button at the bottom left of the codespace window to send a link to other people to join your coding session, kind of like with google docs. A submission from one person will submit for the entire group!

## Working Remotely VS on Your Machine

You normally will be working on this with the github codespaces, but you can also git clone this repository to your own machine and work on it there. Committing and pushing as you go. If you do this I suggest you use VScode as it makes the whole process of cloning very simple, just a few button presses. In a new VScode window go to file browser and hit "Clone Repository," then paste in the link to this repository into the popup. You may need to add a few pip installs but this whole process is pretty straightforward. It is worth noting that if you choose to edit it this way you will need to get the connectoid data a different way than who it is done for the codespace environment. In this [google drive](https://drive.google.com/drive/folders/1r9X9KDWgY0HSV3NPekrjVINlG4XsmYmi) you will find the data set, download this and add it to your personal repository (do not commit it, as it is too big for github), then go to the data loader functions and change the paths to match the new data set source. 

## Testing the Game

You can run the file food_land.py on your local computer. This will allow you to see the environment your organoid will be navigating and might make devising the encode/decode functions easier. Also it is interesting to try to get the game working on your own!

## Getting Started

To begin or resume your assignment, simply click the "Code" button in the top right of your repository area. From there press the + button in the codespaces tab, setting up your codespace may take a few minutes. If you leave after setting the codespace up, you can resume it at any time through the same code menu. Once the codespace is set up make the changes you need, answer the questions and then you can submit in the Source Control tab, by staging your changes with the + button and then using the big green "commit" button with a commit message and then hitting the "Sync Changes" button that pops up afterwards. Feel free to submit as many times as you wish.