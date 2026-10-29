import {chromium} from '@playwright/test';
import {spawn} from 'node:child_process';
import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
const root=path.resolve('.');const out=path.join(root,'docs/assets/v0.3/final');await fs.mkdir(out);
const server=spawn('python3',['-m','http.server','9470','--bind','127.0.0.1','--directory',path.join(root,'lab')],{stdio:'ignore'});
let browser;
try {
  for(let i=0;i<100;i++){try{if((await fetch('http://127.0.0.1:9470/workload.html')).ok)break;}catch{}await new Promise(r=>setTimeout(r,200));}
  browser=await chromium.launch({executablePath:'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',headless:true});
  const context=await browser.newContext({viewport:{width:1440,height:1000},reducedMotion:'reduce'});
  const page=await context.newPage();await page.goto('http://127.0.0.1:9470/workload.html');await page.locator('#evidence-status').waitFor({state:'visible'});await page.waitForFunction(()=>document.querySelector('#evidence-status').textContent==='measured null');await page.evaluate(()=>document.fonts.ready);
  await page.screenshot({path:path.join(out,'workload-desktop.png')});
  await page.locator('#run').screenshot({path:path.join(out,'workload-application.png')});
  await page.setViewportSize({width:390,height:844});await page.goto('http://127.0.0.1:9470/workload.html');await page.waitForFunction(()=>document.querySelector('#evidence-status').textContent==='measured null');await page.evaluate(()=>document.fonts.ready);await page.screenshot({path:path.join(out,'workload-mobile.png')});
  await page.locator('#guard-example').selectOption('identity');await page.locator('#inspect').screenshot({path:path.join(out,'workload-mobile-guards.png')});
  await context.close();
  const videoContext=await browser.newContext({viewport:{width:1440,height:900},recordVideo:{dir:path.join(root,'.work/workload-recording'),size:{width:1440,height:900}}});
  const film=await videoContext.newPage();await film.goto('http://127.0.0.1:9470/workload.html');await film.waitForFunction(()=>document.querySelector('#evidence-status').textContent==='measured null');await film.evaluate(()=>document.fonts.ready);await film.waitForTimeout(1500);
  await film.locator('#install').scrollIntoViewIfNeeded();await film.getByRole('button',{name:'Copy installation commands'}).click();await film.waitForTimeout(1400);
  await film.locator('#consumer').selectOption('vectors');await film.locator('#run').scrollIntoViewIfNeeded();await film.waitForTimeout(1700);await film.locator('.code-disclosure summary').filter({hasText:'Use the callable in Python'}).click();await film.waitForTimeout(1700);
  await film.locator('#recorded-demo summary').click();await film.locator('#recorded-demo').scrollIntoViewIfNeeded();await film.waitForTimeout(2200);await film.locator('#recorded-query').selectOption('1');await film.waitForTimeout(2200);await film.locator('#inspect').scrollIntoViewIfNeeded();await film.locator('#guard-example').selectOption('identity');await film.waitForTimeout(1900);await film.locator('#methods-details summary').click();await film.locator('.measurement-section').scrollIntoViewIfNeeded();await film.waitForTimeout(3000);
  const video=film.video();await film.close();await videoContext.close();const videoPath=await video.path();await fs.copyFile(videoPath,path.join(out,'workload-interaction.webm'));
  const hashes={};for(const name of ['workload.html','workload.css','workload.js','data/workload.json','data/workload-demo.json'])hashes[name]=crypto.createHash('sha256').update(await fs.readFile(path.join(root,'lab',name))).digest('hex');
  await fs.writeFile(path.join(out,'capture.json'),JSON.stringify({kind:'actual_browser_interaction_recording',performance_claim:false,note:'Actual browser interaction with read-only recorded installed GPU CLI outputs and retained null performance study. No live GPU browser execution.',source_sha256:hashes,browser:browser.version(),captured_at:new Date().toISOString()},null,2));
  console.log(JSON.stringify({output:out,video:videoPath,source_sha256:hashes}));
}finally{if(browser)await browser.close();server.kill('SIGTERM');}
