import { OutputParams } from './params';

const generate = () =>
  fetch('http://127.0.0.1:5000/generate')
    .then((response) => response.json())
    .then((response) => JSON.parse(response) as OutputParams);

export default generate;
