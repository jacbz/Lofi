import { TrackFile } from "./TrackFile";
import { TrackInput } from "./TrackInput";
import * as drumFillsTracks from "./files/DrumFills.json";
import * as drumLoopsTracks from "./files/DrumLoops.json";
import * as fullDrumLoopsTracks from "./files/FullDrumLoops.json";
import * as hiHatLoopsTracks from "./files/HiHatLoops.json";
import * as percussionLoopsTracks from "./files/PercussionLoops.json";
import * as topLoopsTracks from "./files/TopLoops.json";

export class Trackfinder {

    drumFills: Array<TrackFile> = drumFillsTracks;

    drumLoopsTracks: Array<TrackFile> = drumLoopsTracks;

    fullDrumLoops: Array<TrackFile> = fullDrumLoopsTracks;

    hiHatLoops: Array<TrackFile> = hiHatLoopsTracks;

    percussionLoops: Array<TrackFile> = percussionLoopsTracks;

    toploops: Array<TrackFile> = topLoopsTracks

    searchTracks(trackInput: TrackInput): Array<TrackFile> {
        const trackFiles = new Array<TrackFile>();

        // Selecting appropriate sound files

        return trackFiles;
    }
}
