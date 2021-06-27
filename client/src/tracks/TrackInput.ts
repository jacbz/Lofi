export class TrackInput {

    private acousticness: number;

    private danceability: number;

    private energy: number;

    private instrumentalness: number;

    private valence: number;

    private liveness: number;

    private speechiness: number;

    public constructor(acousticness: number, danceability: number, energy: number, instrumentalness: number, valence: number, liveness: number, speechiness: number) {
        this.setAcousticness(acousticness);
        this.setDanceability(danceability);
        this.setEnergy(energy);
        this.setInstrumentalness(instrumentalness);
        this.setValence(valence);
        this.setLiveness(liveness);
        this.setSpeechiness(speechiness);
    }

    public getAcousticness(): number {
        return this.acousticness;
    }

    public setAcousticness(acousticness: number): void {
        if (acousticness >= 0 && acousticness <=1) {
            this.acousticness = acousticness;
        } else {
            throw new RangeError();
        }
    }

    public getDanceability(): number {
        return this.danceability;
    }

    public setDanceability(danceability: number): void {
        if (danceability >= 0 && danceability <=1) {
            this.danceability = danceability;
        } else {
            throw new RangeError();
        }
    }

    public getEnergy(): number {
        return this.energy;
    }

    public setEnergy(energy: number): void {
        if (energy >= 0 && energy <=1) {
            this.energy = energy;
        } else {
            throw new RangeError();
        }
    }

    public getInstrumentalness(): number {
        return this.instrumentalness;
    }

    public setInstrumentalness(instrumentalness: number): void {
        if (instrumentalness >= 0 && instrumentalness <=1) {
            this.instrumentalness = instrumentalness;
        } else {
            throw new RangeError();
        }    
    }

    public getValence(): number {
        return this.valence;
    }

    public setValence(valence: number): void {
        if (valence >= 0 && valence <=1) {
            this.valence = valence;
        } else {
            throw new RangeError();
        }   
    }

    public getLiveness(): number {
        return this.liveness;
    }

    public setLiveness(liveness: number): void {
        if (liveness >= 0 && liveness <=1) {
            this.liveness = liveness;
        } else {
            throw new RangeError();
        }   
    }

    public getSpeechiness(): number {
        return this.speechiness;
    }

    public setSpeechiness(speechiness: number): void {
        if (speechiness >= 0 && speechiness <=1) {
            this.speechiness = speechiness;
        } else {
            throw new RangeError();
        }       
    }
    
}