import { OutputParams } from './params';

const generate = (): Promise<OutputParams> =>
  fetch('https://lofiserver.jacobzhang.de/generate')
    .then((response) => response.json())
    .then((response) => JSON.parse(response) as OutputParams);

export default generate;
