import { OutputParams } from './params';

export const generate = (): Promise<OutputParams> =>
  fetch('https://lofiserver.jacobzhang.de/generate')
    .then((response) => response.json())
    .then((response) => JSON.parse(response) as OutputParams);

export const decode = (inputList: number[]): Promise<OutputParams> =>
  fetch(`https://lofiserver.jacobzhang.de/decode?input=${JSON.stringify(inputList)}`)
    .then((response) => response.json())
    .then((response) => JSON.parse(response) as OutputParams);
