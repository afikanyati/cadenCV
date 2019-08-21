# cadenCV

cadenCV is an optical music recognition system written in the Python programming language which reads sheet music and sequences a MIDI file for user playback.

![alt text](https://github.com/anyati/cadenCV/blob/master/resources/README/image1.jpg)

## I. Dependencies

For OMR:
- *Python 3.6*
- *python-numpy*
- *Python matplotlib*
- *python-opencv*
- *Python MIDIUtil*

## II. Usage

To run the OMR system, run the following command in the terminal:

    $ python main.py "image"

where *image* is a placeholder for the sheet music bitmap file. (Accepts *.jpg* and *.png*)

## III. Introduction

Music has long been an integral part of human culture, and today this still remains true. As a form of artistic ex- pression and emotional communication in contemporary society, it’s almost impossible to go a day without hearing a song played or a tune being whistled in your vicinity.

Over the last century, we’ve seen a significant global increase in the use of digital devices, and the production and consumption of digital information - a trend that has left no industry unaffected. The music industry in specific has experienced significant disruption due to digitization - losing control over content distribution, consumption and the sales associated with the two. Digital tools have also ushered new potential for musicians to express themselves in unique ways using synthesizers and voice-distortion technology.

While much has been explored in context of music production, and listener consumption, limited development has been made towards the digitization of pre-information age and printed music documents. To be more specific, today there exists no robust tools for computer perception of music notation, inhibiting the improvement of music storage, playback and manipulation in a digital form. For example, a musician learning a new song would greatly benefit from hearing its auditory representation prior to parsing its notated sheet music. This is the focus of Optical Music Recognition.

## IV. Approach

Music notation comprises a large finite alphabet used to express ways of interpreting and communicating musical ideas. Each character serves to create a visual manifestation of the interrelated properties of musical sound such as pitch, dynamics, time, and timbre. For background, music notes are presented on a staff, a set of five lines each possessing its own pitch. A clef serves to communicate the pitch associated with each of these five lines. In this way, the pitch of a given note is determined by its vertical placement on the staff i.e. it's pitch is consistent with the pitch of the staff line or space on which it rests. The temporal nature of music is represented by the horizontal traversal of the staff. Thus music notation presents information in two dimensions. Certain subsets of the alphabet serve to communicate note duration, relative durations of silence, temporary changes in pitch, dynamics (loudness), and note articulations (changes in musical sound). In practice, certain symbols have almost unlimited variations in representation. As a result, building a computer vision system that understands musical notation is a difficult, but achievable goal.

![alt text](https://github.com/anyati/cadenCV/blob/master/resources/README/image2.jpg)
Typical music primitives used in sheet music.

![alt text](https://github.com/anyati/cadenCV/blob/master/resources/README/image3.jpg)
Bi-dimensional structure of the music staff.

## V. System Scope

For the purpose of this project, I decided to restrict the scope of the alphabet space considered. Consequently, my research sought to develop a OMR system capable of recognizing and representing high resolution sheet music written for a single monophonic musical instrument using note or rest values equal to or greater than eighth notes, expressed on a staff consisting of either the treble or the bass clef and one of the common time signatures, i.e. common time, 2/4, 4/4, 3/4 or 6/8. Consequently, my recognition system cannot perform *key* or *time signature alterations*, or *detect tempo demarcations*, *harmony*, *multi-instrumentation*, *braced-joined staffs*, *tuplets*, *repeats*, *slurs* and *ties*, *articulation* and *dynamic marks*, or *dotted rhythms*.

![alt text](https://github.com/anyati/cadenCV/blob/master/resources/README/image4.jpg)
Standard input to OMR system.


## VI. System Framework

Over time, a general framework for OMR systems has been developed. The framework breaks down the overarching recognition problem into four simpler operations that aid in the recognition, representation and storage of musical documents in a machine-readable format. The framework can be summarized as follows:

1. Image Processing
2. Music Primitive Detection
3. Music Primitive Classification
4. Semantic Reconstruction

The main goal of the preprocessing stage is to modify the original music document to improve the robustness and efficiency of the recognition process. This typically includes *enhancement*, *blurring*, *noise removal*, *morphological operations*, *deskewing* and *binarization*. The music primitive detection stage is where elementary music primitives are recognized, which is followed by the music primitive classification stage where these primitives are classified. Finally, the semantic reconstruction stage reestablishes the semantic relation between the recognized musical primitives and stores the information in a suitable data structure.

### Preprocessing

Because my OMR system was designed for input of a specified nature, namely original printed sheet music recorded in high resolution image file formats, only a few preprocessing operations were necessary. My system employs noise removal, and binarization. The first operation attempts to remove any image artifacts, while the second operation converts the input image into a matrix of binary-valued pixels. Because no information is conveyed through the use of color on common music scores, the process does not constrain subsequent phases.  On the contrary, it facilitates them by reducing the volume of information that is need to be processed. My system employs *Otsu's global thresholding method* for this purpose. A global thresholding method was chosen over an adaptive one due to the fact that the input image would have a uniform background, thus it would not need to threshold each pixel using local neighborhood information.

### Staff Detection

Owing to the importance of staff as a two dimensional coordinate system for musical notation, the first step of primitive detection requires determining *staff line thickness* and *staff spacing height*, which are further used to deduce the size of other music symbols, and to detect the location of each staff in the input image. *Staff line thickness* and *staff spacing height* are calculated using run-length encoding (RLE), by encoding the score column by column with RLE. The most common black run approximates the *staff line thickness* and the most common white run estimates the *staff space height*. Each staff is located using horizontal projection, which maps a binary image into a histogram by accumulating the number of black pixels in each row. The staff is detected as five consequent distinct peaks in the histogram.

![alt text](https://github.com/anyati/cadenCV/blob/master/resources/README/image5.jpg)
The horizontal projection of a music score excerpt.

### Music Primitive Detection and Classification

Music primitives are both segmented and classified using a template matching approach to image classification.The classification method detects primitives by comparing an image segment to a library of labelled primitive templates, staff-by-staff, primitive by primitive, and classifies as a specific music primitive if its similarity exceeds a specified threshold. Because staff positions have been determined beforehand, along with their *staff line thickness* and *staff space height*, note pitch values can be determined by observing which row index the center of a note's bounding box lies; it's pitch is consistent with the pitch of the staff line or space sharing that particular row index.

Eighth notes, which are originally classified as quarter notes owing to the possession of an equivalent note head, are retroactively corrected by determining if an eighth flag is in the vicinity of the note head or whether it is beamed with adjacent notes. Beaming is determined by determining if the column central to adjacent notes contains more black pixels than expected, i.e. more then five times the *staff line thickness*. I had to devise a way to detect key signatures; because every staff is initiated by its key signature, it could easily be identified by counting the number of accidentals located at the beginning of the ordered list of detected music primitives on a given staff. Subsequent to this operation, accidentals resulting from the key are applied to the note primitives.

### Semantic Reconstruction

Once all primitives are located and classified, they are sorted by their horizontal position on their staff and sequenced in order into an object-oriented data structure comprising of *notes*, *rests*, and *accidentals* encapsulated by *bars*, which are further encapsulated by *staffs*. Finally, once each staff has been semantically reconstructed, they are expanded and each note is converted from its note name into a number corresponding to its associated MIDI note value and added to a MIDI track representing the image's auditory representation.

![alt text](https://github.com/anyati/cadenCV/blob/master/resources/README/image6.jpg)
Hierarchy of OMR system's tree data structure.

![alt text](https://github.com/anyati/cadenCV/blob/master/resources/README/image7.jpg)
Staff Detection on 'Mary Had a Little Lamb' input example

![alt text](https://github.com/anyati/cadenCV/blob/master/resources/README/image8.jpg)
Music Primitive Detection and Classification on 'Mary Had a Little Lamb' input example

## VII. Project Paper

To learn more about the cadenCV: Optical Music Recognition system, read the accompanying paper: http://bit.ly/2lfe8Gv

## VII. Output Example

To hear an output example from cadenCV (Mary had a Little Lamb), visit: https://www.youtube.com/watch?v=amL6wHfAShw
