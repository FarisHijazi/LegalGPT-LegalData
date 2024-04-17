
var data = await fetch("https://new.najiz.sa/applications/landing/api/FaqQuestion/GetQuestions", {
  "headers": {
    "accept": "application/json, text/plain, */*",
    "accept-language": "en-US,en;q=0.9",
    "sec-ch-ua": "\"Microsoft Edge\";v=\"123\", \"Not:A-Brand\";v=\"8\", \"Chromium\";v=\"123\"",
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": "\"Windows\"",
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-origin",
  },
  "referrerPolicy": "no-referrer",
  "body": null,
  "method": "GET",
  "mode": "cors",
  "credentials": "include"
}).then(r=>r.json())

console.log(JSON.stringify(data, null, 4));

data.map(z=>z.categoryId[0])

function countElements(array) {
    const count = {};
    array.forEach(item => {
        if (count[item]) {
            count[item] += 1;
        } else {
            count[item] = 1;
        }
    });
    return count;
}

id2count = countElements(data.map(z=>z.categoryId[0]));

function createMapFromList(list) {
    const resultMap = {};
    list.forEach(item => {
        const match = item.match(/(.*) \((\d+)\)/); // Regular expression to extract text and number
        if (match) {
            const text = match[1].trim(); // Text before the parenthesis
            const number = parseInt(match[2], 10); // Number inside the parenthesis
            resultMap[number] = text; // Mapping number to text
        }
    });
    return resultMap;
}


var elements = [...document.querySelectorAll("#app > div > main > div > div.np5QWqaegwKh1ehMEQFY > div.container.BaXlxao_pwsMJbVpJp1o.d-flex.align-start > div.ku3n5R4SrF0vl2xqELxg > button")].slice(1).map(div=>div.innerText);

var count2label = createMapFromList(elements)

const id2label = {};
for (const [id, count] of Object.entries(id2count)) {
    id2label[id] = count2label[count]; // Fixed to map directly using id
}

console.log(JSON.stringify(id2label, null, 4));


function anchorClick(url, name = '', target = 'self') {
    name = name || nameFile(url) || 'filename';

    var a = document.createElement('a');
    a.setAttribute('href', url);
    a.setAttribute('download', name);
    a.setAttribute('target', target);
    document.documentElement.appendChild(a);
    // call download
    // a.click() or CLICK the download link can't modify filename in Firefox (why?)
    // Solution from FileSaver.js, https://github.com/eligrey/FileSaver.js/
    a.dispatchEvent(new MouseEvent('click'));
    document.documentElement.removeChild(a);
}
function makeFile(text, options = {name: false, type: 'text/plain', replace: false}) {
    if (typeof file === 'undefined') {
        var file = null;
    }
    if (typeof (options) === 'string') options = {name: String(options)};
    options = Object.assign({name: false, type: 'text/plain', replace: false}, options);

    const data = new Blob([text], {type: options.type});
    // If we are replacing a previously generated file we need to manually revoke the object URL to avoid memory leaks.
    if (file !== null && options.replace === false) window.URL.revokeObjectURL(file);
    file = window.URL.createObjectURL(data);
    if (options.name) {
        anchorClick(file, options.name);
    }
    return file;
}


makeFile(JSON.stringify(data, null, 4), 'Najiz_QA_data.json')
makeFile(JSON.stringify(id2label, null, 4), 'Najiz_QA_id2label.json')
