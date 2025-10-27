const si = require('systeminformation');
(async () => {
  try {
    const cpu = await si.currentLoad();
    const mem = await si.mem();
    const gpu = await si.graphics();
    console.log(JSON.stringify({ cpu, mem, gpu }, null, 2));
  } catch (error) {
    console.error(error);
    process.exit(1);
  }
})();
